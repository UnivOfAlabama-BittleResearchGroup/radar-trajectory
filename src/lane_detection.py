from __future__ import annotations

from typing import Generator, List, Tuple
from collections import deque


from matplotlib import pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
from scipy.ndimage import (
    gaussian_filter,
    percentile_filter,
)
from skimage.feature.peak import peak_local_max
from sklearn.neighbors import KDTree
import networkx as nx

from .filtering import timeit


# get the distance between two points
def _get_distance(start: Tuple[float, float], end: Tuple[float, float]) -> float:
    return np.linalg.norm(np.array(start) - np.array(end))


def _get_angle(edge1: Tuple[float, float], edge2: Tuple[float, float]) -> float:
    ang1 = np.arctan2(edge1[1][1] - edge1[0][1], edge1[1][0] - edge1[0][0])
    ang2 = np.arctan2(edge2[1][1] - edge2[0][1], edge2[1][0] - edge2[0][0])
    return np.rad2deg((ang1 - ang2 + np.pi) % (2 * np.pi) - np.pi)


class LaneDetection:
    """
    This class is used to detect lanes in a radar "image".
    """

    ANGLE_THRESHOLD = 35

    def __init__(self, df: pl.DataFrame, angle_threshold: float = ANGLE_THRESHOLD):
        self._angle_threshold = angle_threshold

        # dataframe stores
        self._df: pl.DataFrame = df
        self._snapped_df: pl.DataFrame = None
        self._peak_df: pd.DataFrame = None

        # store bin edges
        self._x_bins: np.ndarray = np.array([])
        self._y_bins: np.ndarray = np.array([])

        # store the histogram
        self._hist: np.ndarray = np.array([])
        self._p_hist: np.ndarray = np.array([])
        self._p_hist_smooth: np.ndarray = np.array([])

        # make a kd tree
        self._kd_tree: KDTree = None

        # store the graph
        self._graph: nx.DiGraph = None
        self._birth_nodes: list = []
        self._death_nodes: list = []

        # make an angle cache
        self._angle_cache: dict = {}

    @timeit
    def bin_data(self, bin_size_m: float = 0.1) -> LaneDetection:
        """
        This function bins the data in the dataframe.
        """
        # create the bins
        x_bins = np.arange(
            self._df["f32_positionX_m"].min(),
            self._df["f32_positionX_m"].max(),
            bin_size_m,
        )
        y_bins = np.arange(
            self._df["f32_positionY_m"].min(),
            self._df["f32_positionY_m"].max(),
            bin_size_m,
        )

        self._hist, self._x_bins, self._y_bins = np.histogram2d(
            self._df["f32_positionX_m"],
            self._df["f32_positionY_m"],
            bins=[x_bins, y_bins],
        )
        return self

    @timeit
    def percentile_filter(self, percentile: float = 80) -> LaneDetection:
        """
        This function applies a percentile filter to the histogram.
        """
        self._p_hist = percentile_filter(self._hist, percentile, size=3)
        return self

    @timeit
    def gaussian_filter(
        self, sigma: float = 1.0, sigma_truncate: float = 2
    ) -> LaneDetection:
        """
        This function applies a gaussian filter to the histogram.
        """
        self._p_hist_smooth = gaussian_filter(
            self._p_hist, sigma=sigma, truncate=sigma_truncate
        )
        return self

    @timeit
    def find_peaks(
        self, min_distance: int = 5, threshold_quantile: float = 0.92
    ) -> LaneDetection:
        """
        This function finds the peaks in the histogram.
        """
        peaks = peak_local_max(
            self._p_hist_smooth,
            min_distance=min_distance,
            exclude_border=True,
            threshold_abs=np.quantile(self._p_hist_smooth, threshold_quantile),
        )

        self._peaks = np.array([self._x_bins[peaks[:, 0]], self._y_bins[peaks[:, 1]]]).T
        return self

    def _make_kd_tree(
        self,
    ):
        self._kd_tree = KDTree(self._peaks)

    @timeit
    def snap_df_to_grid(
        self,
        max_dist_m: float = 3.0,
    ) -> LaneDetection:
        if self._kd_tree is None:
            self._make_kd_tree()

        # get the closest grid point for each point in the dataframe
        dist, ind = self._kd_tree.query(
            self._df[["f32_positionX_m", "f32_positionY_m"]].to_numpy()
        )
        closest_peak_points = self._peaks[ind.flatten()]

        # add the grid point to the dataframe
        self._snapped_df = (
            self._df.with_columns(
                [
                    pl.Series("closest_peak_x", closest_peak_points[:, 0]),
                    pl.Series("closest_peak_y", closest_peak_points[:, 1]),
                    pl.Series("dist_to_peak", dist.flatten()),
                ]
            )
            .filter(pl.col("dist_to_peak") < max_dist_m)
            .drop(["dist_to_peak"])
            .with_columns(
                [
                    pl.col("closest_peak_x")
                    .round(2)
                    .cast(pl.Float32)
                    .alias("closest_peak_x"),
                    pl.col("closest_peak_y")
                    .round(2)
                    .cast(pl.Float32)
                    .alias("closest_peak_y"),
                ]
            )
            .groupby(["object_id", "closest_peak_x", "closest_peak_y"])
            .agg(
                pl.col("epoch_time").first().alias("epoch_time"),
            )
            .sort(["object_id", "epoch_time"])
            .with_columns(
                [
                    pl.col("closest_peak_x")
                    .shift(-1)
                    .forward_fill()
                    .over("object_id")
                    .alias("next_x"),
                    pl.col("closest_peak_y")
                    .shift(-1)
                    .forward_fill()
                    .over("object_id")
                    .alias("next_y"),
                ]
            )
            .with_columns(
                [
                    # calculate the y_diff
                    (pl.col("next_y") - pl.col("closest_peak_y")).alias("y_diff"),
                    # calculate the x_diff
                    (pl.col("next_x") - pl.col("closest_peak_x")).alias("x_diff"),
                ]
            )
        )

        vals = self._snapped_df.select(["y_diff", "x_diff"]).to_numpy()
        angle = np.rad2deg(np.arctan2(vals[:, 0], vals[:, 1]))

        self._snapped_df = (
            self._snapped_df.with_columns(
                [
                    pl.Series("angle", angle),
                ]
            )
            .with_columns(
                [
                    pl.col("angle")
                    .shift(-1)
                    .forward_fill()
                    .over("object_id")
                    .alias("next_angle"),
                ]
            )
            .with_columns(
                [
                    # calculate the angle difference
                    ((pl.col("next_angle") - pl.col("angle")) + 180 % 360 - 180).alias(
                        "angle_diff"
                    ),
                ]
            )
            .filter(pl.col("angle_diff").abs() < self._angle_threshold)
        )

        return self

    @timeit
    def create_transition_df(
        self, valid_radius_m: float = 20, filter_count_quantile: int = 60
    ) -> LaneDetection:
        if self._kd_tree is None:
            self._make_kd_tree()

        nn = self._kd_tree.query_radius(self._peaks, r=valid_radius_m)
        matches = np.array(
            [[self._peaks[i], self._peaks[j]] for i, js in enumerate(nn) for j in js]
        ).round(2)

        peak_df = pl.DataFrame(
            {
                "pairs": matches,
                "start": matches[:, 0],
                "end": matches[:, 1],
                "start_x": matches[:, 0, 0],
                "start_y": matches[:, 0, 1],
                "end_x": matches[:, 1, 0],
                "end_y": matches[:, 1, 1],
            }
        ).with_columns([pl.col(pl.FLOAT_DTYPES).cast(pl.Float32)])

        # join the peak dataframe to the snapped dataframe
        self._peak_df = (
            peak_df.lazy()
            .join(
                self._snapped_df.select(
                    [
                        "closest_peak_x",
                        "closest_peak_y",
                        "next_x",
                        "next_y",
                        "object_id",
                    ]
                ).lazy(),
                left_on=["start_x", "start_y", "end_x", "end_y"],
                right_on=["closest_peak_x", "closest_peak_y", "next_x", "next_y"],
                how="inner",
                suffix="_start",
            )
            .groupby(["start_x", "start_y", "end_x", "end_y"])
            .agg(
                [
                    pl.col("object_id").count().cast(pl.Int32).alias("count"),
                    pl.col("start").first().alias("start"),
                    pl.col("end").first().alias("end"),
                ]
            )
            .filter((pl.col("count") > pl.col("count").quantile(0.6)))
            .collect()
            .to_pandas()
        )

        # turn the start and end into tuples
        self._peak_df[["start", "end"]] = self._peak_df[["start", "end"]].applymap(
            lambda x: tuple(np.round(x, 2).tolist())
        )
        return self

    @timeit
    def make_graph(
        self,
    ) -> LaneDetection:
        self._graph = nx.from_pandas_edgelist(
            self._peak_df,
            source="start",
            target="end",
            edge_attr="count",
            create_using=nx.DiGraph,
        )

        # update the weights
        (
            self._update_weights()
            ._add_distance()
            .drop_bidirectional_edges()
            ._make_angle_cache()
        )
        return self

    @timeit
    def _make_angle_cache(
        self,
    ) -> LaneDetection:
        self._angle_cache = {}

        # walk the nodes and their in/out edges
        for node in self._graph.nodes:
            for in_edge in self._graph.in_edges(node, default=[]):
                for out_edge in self._graph.out_edges(node, default=[]):
                    self._angle_cache[(in_edge, out_edge)] = abs(
                        _get_angle(in_edge, out_edge)
                    )
        return self

    @timeit
    def _update_weights(
        self,
    ) -> LaneDetection:
        for node in self._graph.nodes:
            self._graph.nodes[node]["weight"] = sum(
                n[-1] for n in self._graph.in_edges(node, data="count")
            )
            if self._graph.nodes[node]["weight"] == 0:
                self._graph.nodes[node]["weight"] = sum(
                    n[-1] for n in self._graph.out_edges(node, data="count")
                )
        return self

    @timeit
    def _add_distance(
        self,
    ) -> LaneDetection:
        for edge in self._graph.edges:
            self._graph.edges[edge]["distance"] = _get_distance(edge[0], edge[1])

        return self

    @timeit
    def drop_bidirectional_edges(
        self,
    ) -> LaneDetection:
        remove_edges = set()

        for edge in self._graph.edges:
            if self._graph.has_edge(edge[1], edge[0]):
                # remove_edges.add(edge)
                if (
                    self._graph.edges[edge]["count"]
                    > self._graph.edges[(edge[1], edge[0])]["count"]
                ):
                    remove_edges.add((edge[1], edge[0]))
                else:
                    remove_edges.add(edge)

        self._graph.remove_edges_from(remove_edges)

        return self

    @timeit
    def clean_subgraphs(self, cutoff_size_nodes: int = 10) -> LaneDetection:
        subgraphs = list(nx.weakly_connected_components(self._graph))

        # sort the subgraphs by size
        valid_subgraphs = [s for s in subgraphs if len(s) > cutoff_size_nodes]

        # remove the nodes that are not in the largest subgraph
        self._graph = self._graph.subgraph(n for v in valid_subgraphs for n in v).copy()

        return self

    @timeit
    def clean_isolated_nodes(self) -> LaneDetection:
        isolated_nodes = list(nx.isolates(self._graph))
        self._graph.remove_nodes_from(isolated_nodes)
        return self

    @timeit
    def clean_graph_outs(
        self,
    ) -> LaneDetection:
        # get nodes that have no outgoing edges
        nodes = [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]

        # out edges should only have 1 incoming edge
        for node in nodes:
            if in_edges := list(self._graph.in_edges(node)):
                max_edge = max(in_edges, key=lambda x: self._graph.edges[x]["count"])
                if len(in_edges) > 1:
                    # keep only max edge
                    # remove the other edges
                    for edge in in_edges:
                        if edge != max_edge:
                            self._graph.remove_edge(*edge)
                
                # # also, the node's in edge should come from a node with only 1 out edge (the node)
                if self._graph.out_degree(max_edge[0]) > 1:
                    # remove the node and its edges
                    self._graph.remove_edge(*max_edge)
                    self._graph.remove_node(node)


        return self

    @timeit
    def clean_graph_ins(
        self,
    ) -> LaneDetection:
        # get nodes that have no incoming edges
        nodes = [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]

        # in edges should only have 1 outgoing edge
        for node in nodes:
            if out_edges := list(self._graph.out_edges(node)):
                max_edge = max(out_edges, key=lambda x: self._graph.edges[x]["count"])
                if len(out_edges) > 1:
                    # keep only max edge
                    # remove the other edges
                    for edge in out_edges:
                        if edge != max_edge:
                            self._graph.remove_edge(*edge)

                # # also, the node's out edge should lead to a node with only 1 in edge (the node)
                if self._graph.in_degree(max_edge[1]) > 1:
                    # remove the node and the edge
                    self._graph.remove_edge(*max_edge)
                    self._graph.remove_node(node)

        return self

    @timeit
    def get_birth_nodes(
        self,
    ) -> LaneDetection:
        self._birth_nodes = [
            n for n in self._graph.nodes if self._graph.in_degree(n) == 0
        ]
        return self

    @timeit
    def get_death_nodes(
        self,
    ) -> LaneDetection:
        self._death_nodes = [
            n for n in self._graph.nodes if self._graph.out_degree(n) == 0
        ]
        return self

    def _get_path(self, a: tuple, b: tuple) -> Generator[tuple, None, None]:
        try:
            path = nx.shortest_path(
                self._graph, a, b, method="dijkstra", weight="distance"
            )
        except nx.NetworkXNoPath:
            return ([], 0, 0)

        # split the path every time there is a movement with a large angle
        split_path = [[path[0]]]
        for i in range(len(path) - 2):
            angle = self._angle_cache[
                ((path[i], path[i + 1]), (path[i + 1], path[i + 2]))
            ]
            if angle < 30:
                split_path[-1].append(path[i + 1])
            else:
                split_path.append([])

        for p in split_path:
            if len(p) > 1:
                yield (
                    p,
                    sum(
                        self._graph.edges[(p[i], p[i + 1])]["distance"]
                        for i in range(len(p) - 1)
                    ),
                    min(self._graph.nodes[n]["weight"] for n in p),
                )

    @timeit
    def create_routes(
        self,
    ) -> List[Tuple[float, float]]:
        rs = []
        for a in self._birth_nodes:
            for b in self._death_nodes:
                rs.extend(
                    (path, count, weight)
                    for path, count, weight in self._get_path(a, b)
                )
        return rs

    @timeit
    def consolidate_routes(
        self, routes: List[Tuple[float, float]], min_route_length_m: float = 10
    ) -> List[Tuple[float, float]]:
        # sourcery skip: low-code-quality
        # I don't want to change this code, its a bit of a mess
        """
        Convert a list of routes into a list of consolidated routes. Assures that there are no overlapping routes.

        Args:
            routes (List[Tuple[float, float]]): routes to consolidate, from create_routes
            min_route_length_m (float, optional): minimum route length in meters. Defaults to 10.
        Returns:
            List[Tuple[float, float]]: consolidated routes
        """
        routes = sorted(
            [r for r in routes if r[1] >= min_route_length_m],
            key=lambda x: x[2],
            reverse=True,
        )
        i = 0
        while i < len(routes):
            j = 0
            while j < len(routes):
                if (
                    i != j
                    and len(set(routes[j][0]).intersection(set(routes[i][0]))) > 2
                ):
                    new_path = deque()
                    for k, p in enumerate(routes[j][0]):
                        if p not in routes[i][0]:
                            new_path.append((k, p))

                    if len(new_path):
                        # find a continuous path
                        contig_path = [[new_path.popleft()]]
                        while len(new_path):
                            if (new_path[0][0] - contig_path[-1][-1][0]) == 1:
                                contig_path[-1].append(new_path.popleft())
                            else:
                                contig_path.append([new_path.popleft()])

                        # sort contig path by count
                        contig_path = sorted(
                            contig_path, key=lambda x: len(x), reverse=True
                        )
                        # add these paths to the route
                        for p in contig_path:
                            if len(p) >= 2:
                                r = [_p[1] for _p in p]
                                new_r = (
                                    r,
                                    sum(
                                        self._graph.edges[(r[i], r[i + 1])]["distance"]
                                        for i in range(len(r) - 1)
                                    ),
                                    min(self._graph.nodes[n]["weight"] for n in r),
                                )
                                if new_r[1] >= min_route_length_m:
                                    routes.append(new_r)
                    del routes[j]
                else:
                    j += 1
            i += 1

        return [r for r in routes if r[1] >= min_route_length_m]

    @timeit
    def plot_graph(
        self,
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        nodes = self._graph.nodes

        # get the x and y coordinates
        x = [n[0] for n in nodes]
        y = [n[1] for n in nodes]

        # get the size of the nodes
        size = np.log([n[1]["weight"] for n in nodes.items()])

        # plot the nodes
        ax.scatter(x, y, s=size, alpha=0.5)

        # plot the edges
        for edge in self._graph.edges:
            ax.plot(
                [edge[0][0], edge[1][0]],
                [edge[0][1], edge[1][1]],
                color="r",
                alpha=0.5,
                linewidth=max(self._graph.edges[edge]["count"] ** (1 / 5), 1),
            )

        ax.set_aspect("equal")

        return fig, ax

    @timeit
    def plot_routes(
        self,
        routes: List[Tuple[float, float]],
        ax: plt.Axes = None,
        color_scale: list = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        for j, route in enumerate(routes):
            for i in range(len(route[0]) - 1):
                ax.plot(
                    [route[0][i][0], route[0][i + 1][0]],
                    [route[0][i][1], route[0][i + 1][1]],
                    color="r" if color_scale is None else color_scale[j],
                    linewidth=4,
                )

        ax.set_aspect("equal")

        return fig, ax

    def plot_peaks(
        self,
    ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()

        ax.scatter(
            self._peaks[:, 0],
            self._peaks[:, 1],
            c="r",
            s=10,
        )

        ax.set_aspect("equal")

        return fig, ax

    # plot the histograms
    def plot_histogram(self, hist_type: str = "hist") -> Tuple[plt.Figure, plt.Axes]:
        """_summary_

        Args:
            hist_type (str, optional): the internal historgram to plot. Defaults to "hist".
                Other options are "p_hist" and "p_hist_smooth".
        """
        # plot the histogram
        fig, ax = plt.subplots()

        ax.imshow(
            getattr(self, f"_{hist_type}").T,
            origin="lower",
            extent=[
                self._x_bins[0],
                self._x_bins[-1],
                self._y_bins[0],
                self._y_bins[-1],
            ],
            interpolation="nearest",
            norm="log",
            cmap="jet",
        )

        ax.set_aspect("equal")

        return fig, ax
