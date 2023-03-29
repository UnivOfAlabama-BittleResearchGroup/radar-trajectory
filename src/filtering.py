from dataclasses import dataclass
import itertools
from typing import Dict, Iterator, List, Set, Tuple
import numpy as np
import polars as pl
import h3
import json
import utm
from shapely.geometry import Polygon
import math
import scipy.optimize as opt
from scipy.stats import circmean


# create a wrapper function to time the function
def timeit(func):
    def timed(*args, **kwargs):
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        print(
            "function: {} took: {} seconds".format(func.__name__, end_time - start_time)
        )

        return result

    return timed


class Filtering:
    def __init__(
        self,
        radar_location_path: str,
        network_boundary_path: str,
        # overlap_zone_path: str,
    ) -> None:
        self.h3_resolution = 14

        (
            self.radar_locations,
            self.utm_zone,
            self.rotations,
        ) = self._read_radar_locations(radar_location_path)

        self.network_boundary = self._read_network_boundary(network_boundary_path)

    @staticmethod
    def _read_radar_locations(path) -> Tuple[dict, Tuple]:
        with open(path, "r") as f:
            radar_locations = json.load(f)

        radar_utms = {
            ip: utm.from_latlon(*radar_locations[ip]["origin"][::-1])
            for ip in radar_locations
        }

        radar_rotations = {
            ip: radar_locations[ip]["angle"] * (math.pi / 180) for ip in radar_locations
        }

        # assert that all radar utm zones are the same
        assert len({(x[2], x[3]) for x in radar_utms.values()}) == 1

        return radar_utms, list(radar_utms.values())[0][2:4], radar_rotations

    def _read_network_boundary(self, path) -> Set[str]:
        # open the network boundary shapefile with g
        with open(path, "r") as f:
            json_data = json.load(f)

        ls = Polygon(json_data["features"][0]["geometry"]["coordinates"][0])

        return h3.polyfill_polygon(
            ls.exterior.coords,
            self.h3_resolution,
            lnglat_order=True,
        )

    def create_object_id(self, df: pl.DataFrame) -> pl.DataFrame:
        # make the object id the ui32_objectId + the ip + the date
        return df.with_columns(
            [
                (
                    pl.col("ui32_objectID").cast(pl.Utf8)
                    + "~"
                    + pl.col("ip").cast(pl.Utf8)
                    + "~"
                    + pl.col("epoch_time").dt.strftime("%Y-%m-%d")
                ).alias("object_id"),
            ]
        )

    @timeit
    def fix_stop_param_walk(self, df: pl.DataFrame) -> pl.DataFrame:
        ffill_cols = [
            'f32_velocityInDir_mps',
            'utm_x',
            'utm_y',
            'lat',
            'lon',
            'direction',
        ]
                    

        return (
            df.sort("epoch_time")
            .with_columns(
                [
                    (
                        # mark stops (velocity < 0.01 m/s) & last velocity also < 0.01 m/s
                        ((pl.col("f32_velocityInDir_mps").shift(1) < 0.01) & (pl.col("f32_velocityInDir_mps") < 0.01)).fill_null(
                            False
                        )
                    )
                    .over("object_id")
                    .alias("stopped"),
                ]
            )
            .with_columns(
                [
                    pl.when(pl.col('stopped'))
                    .then(
                        pl.lit(None)
                    ).otherwise(
                        pl.col(c)
                    ).alias(c) for c in ffill_cols
                ]
            )
            # forward fill the nulls
            .with_columns(
                [
                    pl.col(c).forward_fill().over('object_id') for c in ffill_cols
                ]
            )
            .drop(
                [
                    'stopped',
                ]
            )
        )

    @timeit
    def clip_trajectory_end(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.sort("epoch_time")
            .with_columns(
                [
                    (
                        (pl.col("f32_velocityInDir_mps").diff().abs() < 0.01).fill_null(
                            True
                        )
                        & (pl.col("f32_velocityInDir_mps") > 0)
                    )
                    .over("object_id")
                    .alias("stopped"),
                ]
            )
            .with_columns(
                [
                    (~pl.col("stopped"))
                    .cast(pl.Int8())
                    .cumsum()
                    .over("object_id")
                    .alias("stopped_count")
                ]
            )
            .with_columns(
                (pl.col("stopped_count") >= pl.col("stopped_count").max())
                .over("object_id")
                .alias("trim")
            )
            .filter(~pl.col("trim"))
            .sort(["object_id", "epoch_time"])
            .drop(["stopped", "stopped_count", "trim"])
        )

    @timeit
    def filter_short_trajectories(
        self,
        df: pl.DataFrame,
        minimum_distance_m: int = 200,
        minimum_duration_s: int = 5,
    ) -> pl.DataFrame:
        return df.filter(
            pl.col("object_id").is_in(
                df.groupby("object_id")
                .agg(
                    [
                        # calculate the distance between the first and last position
                        (
                            (pl.col("utm_x").first() - pl.col("utm_x").last()).pow(2)
                            + (pl.col("utm_y").first() - pl.col("utm_y").last()).pow(2)
                        )
                        .sqrt()
                        .alias("straight_distance"),
                        # calculate the time between the first and last position
                        (pl.col("epoch_time").last() - pl.col("epoch_time").first())
                        .dt.seconds()
                        .alias("duration"),
                    ]
                )
                .filter(
                    (pl.col("straight_distance") >= minimum_distance_m)
                    & (pl.col("duration") >= minimum_duration_s)
                )["object_id"]
                .to_list()
            )
        )

    @timeit
    def resample(
        self, df: pl.DataFrame, resample_interval_ms: int = 100
    ) -> pl.DataFrame:
        assert resample_interval_ms > 0, "resample_interval_ms must be positive"
        assert "object_id" in df.columns, "object_id must be in the dataframe"
        return (
            df.sort("epoch_time")
            .groupby_dynamic(
                index_column="epoch_time",
                every=f"{100}ms",
                by=["object_id"],
            )
            .agg(
                [
                    pl.col(pl.FLOAT_DTYPES).mean(),
                    *(
                        pl.col(type_).first()
                        for type_ in [pl.INTEGER_DTYPES, pl.Utf8, pl.Boolean]
                    ),
                ]
            )
        )

    @timeit
    def rotate_heading(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.struct(["f32_directionX", "f32_directionY", "ip"])
                .apply(
                    lambda x: (
                        np.arctan2(x["f32_directionY"], x["f32_directionX"])
                        - self.rotations[x["ip"]]
                    )
                    % (2 * np.pi),
                )
                .alias("direction"),
            ]
        ).with_columns(
            [
                # add also the direction in degrees
                (pl.col("direction") * (180 / np.pi)).alias("direction_degrees")
            ]
        )

    @timeit
    def correct_center(
        self,
        df: pl.DataFrame,
        x_col: str = "f32_positionX_m",
        y_col: str = "f32_positionY_m",
    ) -> pl.DataFrame:
        return (
            df.with_columns(
                [
                    # precompute the angle sin and cos
                    pl.col("direction").sin().alias("sin"),
                    pl.col("direction").cos().alias("cos"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col(x_col)
                        + (
                            # subtract this to get the font of the car
                            +(pl.col("f32_distanceToBack_m").abs()) * pl.col("cos")
                            # add this to get the center of the car
                            - (pl.col("f32_length_m") / 2 * pl.col("cos"))
                        )
                    ).alias(x_col),
                    (
                        pl.col(y_col)
                        + (
                            # subtract this to get the font of the car
                            +(pl.col("f32_distanceToBack_m").abs()) * pl.col("sin")
                            # add this to get the center of the car
                            - (pl.col("f32_length_m") / 2 * pl.col("sin"))
                        )
                    ).alias(y_col),
                ]
            )
            .drop(["cos", "sin"])
        )

    @timeit
    def rotate_radars(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        return (
            df.with_columns(
                [
                    pl.col("ip")
                    .apply(lambda ip: self.rotations[ip])
                    .alias("rotation_angle"),
                ]
            )
            .with_columns(
                [
                    (
                        pl.col("f32_positionX_m") * pl.col("rotation_angle").cos()
                        + pl.col("f32_positionY_m") * pl.col("rotation_angle").sin()
                    ).alias("rotated_x"),
                    (
                        -pl.col("f32_positionX_m") * pl.col("rotation_angle").sin()
                        + pl.col("f32_positionY_m") * pl.col("rotation_angle").cos()
                    ).alias("rotated_y"),
                ]
            )
            .drop(["rotation_angle"])
        )

    @timeit
    def radar_to_utm(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                (
                    pl.col("rotated_x")
                    + pl.col("ip").apply(lambda ip: self.radar_locations[ip][0])
                ).alias("x"),
                (
                    pl.col("rotated_y")
                    + pl.col("ip").apply(lambda ip: self.radar_locations[ip][1])
                ).alias("y"),
            ]
        )

    @timeit
    def radar_to_latlon(self, df: pl.DataFrame) -> pl.DataFrame:
        # add a row number column
        df = df.with_row_count()
        tmp = df.select(
            [
                "row_nr",
                "x",
                "y",
            ]
        ).to_pandas(use_pyarrow_extension_array=False)

        # convert to latlon using utm. This could be chunked
        # TODO: chunk this
        tmp["lat_new"], tmp["lon_new"] = utm.to_latlon(
            tmp["x"].values, tmp["y"].values, self.utm_zone[0], self.utm_zone[1]
        )

        # convert back to polars
        return df.join(
            pl.from_pandas(tmp[["lat_new", "lon_new", "row_nr"]], include_index=False),
            on="row_nr",
        ).drop(["row_nr"])

    @timeit
    def radar_to_h3(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_column(
            pl.struct(["lat", "lon"])
            .apply(lambda x: h3.geo_to_h3(x["lat"], x["lon"], self.h3_resolution))
            .alias("h3")
        )

    @timeit
    def int_h3_2_str(self, df: pl.DataFrame) -> pl.DataFrame:
        if df["h3"].dtype == pl.Utf8:
            return df

        import h3.api.basic_int as h3_int

        return df.with_columns([pl.col("h3").apply(h3_int.h3_to_string).alias("h3")])

    @timeit
    def filter_network_boundaries(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("h3").is_in(list(self.network_boundary)))


@dataclass
class RadarHandoff:
    to: str
    from_: str

    def __str__(self) -> str:
        return f"{self.from_} -> {self.to}"

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash((self.from_, self.to))

    # make a tuple of the two radars
    def __iter__(self) -> Iterator[str]:
        yield from (self.from_, self.to)

    def __eq__(self, other: object) -> bool:
        return (
            self.from_ == other.from_ and self.to == other.to
            if isinstance(other, RadarHandoff)
            else NotImplemented
        )


class Fusion:
    def __init__(
        self,
        # f: Filtering,
        radar_handoffs: List[RadarHandoff],
        overlap_threshold: float = 0.05,
    ) -> None:
        # self._f = f
        self._radar_pairs: List[RadarHandoff] = radar_handoffs
        self._overlap_threshold: float = overlap_threshold
        self._overlap_h3s: Dict[RadarHandoff, set] = {}

    def build_overlaps(self, df: pl.DataFrame) -> None:
        overlaps = (
            df.groupby(["h3", "ip"])
            .agg(
                [
                    pl.col("object_id").count().alias("count"),
                ]
            )
            .pivot(values="count", index="h3", columns="ip", aggregate_function="sum")
            .fill_null(0)
            .to_pandas()
            .set_index("h3")
        )

        overlaps = overlaps.div(overlaps.sum(axis=1), axis=0)

        search_combinations = list(itertools.permutations(overlaps.columns, 2))

        self._overlap_h3s = {
            pair: list(
                set(overlaps.loc[overlaps[pair[0]] > self._overlap_threshold].index)
                & set(overlaps.loc[overlaps[pair[1]] > self._overlap_threshold].index)
            )
            for pair in search_combinations
        }

        # if the overlap is empty, pop it
        for k, v in list(self._overlap_h3s.items()):
            # TODO: this should probably be a parameter
            if len(v) < 10:
                self._overlap_h3s.pop(k)

    def get_overlaps(
        self,
    ) -> pl.DataFrame:
        return pl.from_dicts(
            [
                {
                    "h3": h3,
                    "from": p[0],
                    "to": p[1],
                }
                # for p in self._radar_pairs
                for p, h3 in self._overlap_h3s.items()
            ]
        )

    @timeit
    def _merge_radar(self, df: pl.DataFrame, target_pair: RadarHandoff) -> pl.DataFrame:
        print("Joining: ", target_pair)

        overlap_df = df.filter(pl.col("h3").is_in(self._overlap_h3s[target_pair]))
        diff_columns = ["utm_x", "utm_y", "f32_velocityInDir_mps", "direction"]

        return (
            overlap_df.filter(pl.col("ip") == target_pair[0])
            .join(
                overlap_df.filter(pl.col("ip") == target_pair[1]).select(
                    [
                        "h3",
                        "epoch_time",
                        "object_id",
                        "utm_x",
                        "utm_y",
                        "f32_velocityInDir_mps",
                        "f32_length_m",
                        "direction",
                    ]
                ),
                on="epoch_time",
                how="inner",
                suffix="_search",
            )
            .sort("epoch_time")
            .with_columns(
                [
                    *(
                        (pl.col(c) - pl.col(c + "_search")).abs().alias(c + "_diff")
                        for c in diff_columns
                    ),
                ]
            )
            # standard scale the data. I don't think this is the best solution, but it works for now
            .with_columns(
                [
                    *(
                        (
                            (pl.col(c) - pl.col(c).min())
                            / (pl.col(c).max() - pl.col(c).min())
                        ).alias(c)
                        for c in map(lambda x: x + "_diff", diff_columns)
                    ),
                ]
            )
            .with_columns(
                [
                    sum(
                        [
                            pl.col(c).pow(2)
                            for c in map(lambda x: x + "_diff", diff_columns)
                        ]
                    )
                    .sqrt()
                    .alias("distance"),
                ]
            )
            .groupby(["object_id_search", "object_id"])
            .agg(
                [
                    # calculate the euclidean distance between the a vection of x, y, and velocity
                    pl.col("distance")
                    .mean()
                    .alias("distance"),
                ]
            )
            .with_columns([pl.col("distance")])
            .filter(pl.col("object_id_search") != pl.col("object_id"))
        )

    @timeit
    def _hungarian_match(self, distance_df: pl.DataFrame) -> pl.DataFrame:
        cost_df = distance_df.pivot(
            values="distance", index="object_id_search", columns="object_id"
        )

        # store the rows (object_id_search) and columns (object_id) for later
        rows = cost_df["object_id_search"]
        columns = [c for c in cost_df.columns if c != "object_id_search"]

        # create a cost matrix
        cost_matrix = (cost_df.fill_null(1e6).select(columns)).to_numpy()

        # get the optimal matching
        row_ind, col_ind = opt.linear_sum_assignment(cost_matrix)

        # create a dataframe of the optimal matching
        return pl.DataFrame(
            {
                "object_id_search": rows.take(row_ind),
                "object_id": [columns[i] for i in col_ind],
                "distance": cost_matrix[row_ind, col_ind],
            }
        )

    @timeit
    def merge_radars(
        self,
        df: pl.DataFrame,
        radar_pair: RadarHandoff = None,
        return_distance: bool = False,
    ) -> pl.DataFrame:
        radar_pairs = (
            [(radar_pair.to, radar_pair.from_)]
            if radar_pair is not None
            else list(self._overlap_h3s.keys())
        )

        if return_distance:
            return pl.concat(
                [(df.pipe(self._merge_radar, pair)) for pair in radar_pairs]
            )
        return pl.concat(
            [
                df.pipe(self._merge_radar, pair).pipe(self._hungarian_match)
                for pair in radar_pairs
            ]
        )
