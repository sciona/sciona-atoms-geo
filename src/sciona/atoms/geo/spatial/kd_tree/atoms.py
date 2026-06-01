"""Spatial and nearest-neighbor query atoms using KD-Trees."""

from __future__ import annotations

import icontract
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree

from sciona.ghost.registry import register_atom

from .witnesses import witness_build_kd_tree, witness_query_kd_tree


def _finite_2d_array(data: NDArray[np.float64]) -> bool:
    arr = np.asarray(data, dtype=np.float64)
    return bool(arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] > 0 and np.all(np.isfinite(arr)))


@register_atom(witness_build_kd_tree)  # type: ignore[untyped-decorator]
@icontract.require(lambda data: _finite_2d_array(data), "data must be a non-empty, finite 2-D array of shape (N, D)")
@icontract.require(lambda leafsize: leafsize >= 1, "leafsize must be a positive integer")
@icontract.ensure(lambda result, data: result.n == data.shape[0] and result.m == data.shape[1], "tree must encapsulate coordinate shapes")
def build_kd_tree(
    data: NDArray[np.float64],
    leafsize: int = 16,
) -> KDTree:
    """Construct a spatial KD-tree index from coordinate data.

    A $k$-dimensional tree (KD-Tree) partitions spatial coordinate points in
    the real space recursively by splitting medians along axes. This allows for
    highly efficient average time spatial query execution, bypassing brute-force
    distance matrix computations.
    """
    arr = np.asarray(data, dtype=np.float64)
    return KDTree(arr, leafsize=leafsize)


@register_atom(witness_query_kd_tree)  # type: ignore[untyped-decorator]
@icontract.require(lambda tree: tree is not None, "tree must be a valid KDTree instance")
@icontract.require(lambda query_points, tree: np.asarray(query_points).shape[-1] == tree.m, "query_points last dimension must match the KDTree spatial dimension")
@icontract.require(lambda k: k >= 1, "k must be at least 1")
@icontract.require(lambda p: p >= 1.0, "p must be at least 1.0 (Minkowski norm parameter)")
@icontract.require(lambda distance_upper_bound: distance_upper_bound > 0.0, "distance_upper_bound must be positive")
@icontract.ensure(lambda result: result[0].shape == result[1].shape, "distances and indices must share shape")
@icontract.ensure(lambda result: np.all(result[1] >= -1), "indices must be non-negative (or -1 if neighbor not found)")
def query_kd_tree(
    tree: KDTree,
    query_points: NDArray[np.float64],
    k: int = 1,
    p: float = 2.0,
    distance_upper_bound: float = float("inf"),
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Query a pre-built KD-tree for k-nearest neighbors.

    Performs space-partitioning search in Minkowski Lp space using cell-to-query
    distance bounds, hypersphere pruning, and backtracking.
    """
    query_arr = np.asarray(query_points, dtype=np.float64)
    distances, indices = tree.query(
        query_arr,
        k=k,
        p=p,
        distance_upper_bound=distance_upper_bound,
    )
    return np.asarray(distances, dtype=np.float64), np.asarray(indices, dtype=np.int64)
