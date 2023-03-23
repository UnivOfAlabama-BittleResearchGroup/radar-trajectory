from dataclasses import dataclass
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

    @timeit
    def rotate_heading(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(
            [
                pl.struct(["f32_directionX", "f32_directionY", "ip"])
                .apply(
                    lambda x: 
                        (np.arctan2(x["f32_directionY"], x["f32_directionX"])
                        - self.rotations[x["ip"]])
                        % (2 * np.pi),
                )
                .alias("direction"),
            ]
        ).with_columns([
            # add also the direction in degrees
            (pl.col("direction") * (180 / np.pi)).alias("direction_degrees")
        ])

    @timeit
    def correct_center(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                [
                    # precompute the angle sin and cos
                    pl.col("direction").sin().alias("sin"),
                    pl.col("direction").cos().alias("cos"),
                ]
            ).with_columns(
                [
                    (
                        pl.col("f32_positionX_m")
                        + (
                            # subtract this to get the font of the car
                            + (pl.col("f32_distanceToBack_m").abs()) * pl.col("cos")
                            # add this to get the center of the car
                            - (pl.col("f32_length_m") / 2 * pl.col("cos"))
                        )
                    ).alias("f32_positionX_m"),
                    (
                        pl.col("f32_positionY_m")
                        + (
                            # subtract this to get the font of the car
                            + (pl.col("f32_distanceToBack_m").abs()) * pl.col("sin")
                            # add this to get the center of the car
                            - (pl.col("f32_length_m") / 2 * pl.col("sin"))
                        )
                    ).alias("f32_positionY_m"),
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

        self._overlap_h3s = {
            pair: list(
                set(overlaps.loc[overlaps[pair.to] > self._overlap_threshold].index)
                & set(
                    overlaps.loc[overlaps[pair.from_] > self._overlap_threshold].index
                )
            )
            for pair in self._radar_pairs
        }

    def get_overlaps(
        self,
    ) -> pl.DataFrame:
        return pl.from_dicts(
            [
                {
                    "h3": h3,
                    "from": p.from_,
                    "to": p.to,
                }
                for p in self._radar_pairs
                for h3 in self._overlap_h3s[p]
            ]
        )

    @timeit
    def _merge_radar(self, df: pl.DataFrame, target_pair: RadarHandoff) -> pl.DataFrame:
        print("Joining: ", target_pair)

        overlap_df = df.filter(pl.col("h3").is_in(self._overlap_h3s[target_pair]))

        # we need to only get vehicles which end closer to the to radar than the from radar. Unfortunately not that simple
        # the code must
        # to_vehicles = df.filter(pl.col("ip") == target_pair.to).groupby(
        #     "object_id"
        # ).agg([
        #     pl.col("h3")
        # ])
        diff_columns = ["utm_x", "utm_y", "f32_velocityInDir_mps", "direction"]

        return (
            overlap_df.filter(pl.col("ip") == target_pair.to)
            .join(
                overlap_df.filter(pl.col("ip") == target_pair.from_).select(
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
                    # force the angle difference to 0 when stationary (this removes the walking that radar does)
                    # pl.when(
                    #     (pl.col("f32_velocityInDir_mps") > 0)
                    #     & (pl.col("f32_velocityInDir_mps_search") > 0)
                    # )
                    # .then((pl.col("angle") - pl.col("angle_search")).abs())
                    # .otherwise(pl.lit(0))
                    # .alias("angle_diff"),
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
                    pl.col("distance").mean().alias("distance"),
                    # (pl.col("utm_y").first() - pl.col("utm_y").last()).alias(
                    #     "utm_y_diff"
                    # ),
                    # (pl.col("utm_x").first() - pl.col("utm_x").last()).alias(
                    #     "utm_x_diff"
                    # ),
                    # (
                    #     pl.col("utm_y_search").first() - pl.col("utm_y_search").last()
                    # ).alias("utm_y_diff_search"),
                    # (
                    #     pl.col("utm_x_search").first() - pl.col("utm_x_search").last()
                    # ).alias("utm_x_diff_search"),
                ]
            )
            # .with_columns(
            #     [
            #         pl.struct(["utm_y_diff_search", "utm_x_diff_search"])
            #         .apply(
            #             lambda x: math.atan2(
            #                 x["utm_y_diff_search"], x["utm_x_diff_search"]
            #             )
            #         )
            #         .alias("search_angle"),
            #         pl.struct(["utm_y_diff", "utm_x_diff"])
            #         .apply(lambda x: math.atan2(x["utm_y_diff"], x["utm_x_diff"]))
            #         .alias("veh_angle"),
            #     ]
            # )
            .with_columns(
                [
                    pl.col("distance")
                    # add in the angle error as percent of circle. multiplied by 10
                    # + (
                    #     (
                    #         (pl.col("search_angle") - pl.col("veh_angle")).abs()
                    #         / (2 * math.pi)
                    #     )
                    #     # * 10
                    # )
                ]
            )
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
        radar_pairs = [radar_pair] if radar_pair is not None else self._radar_pairs

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
