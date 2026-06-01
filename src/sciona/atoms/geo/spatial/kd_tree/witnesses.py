"""Ghost witnesses for KD-Tree spatial indexing and neighbor search atoms."""

from __future__ import annotations

from pydantic import BaseModel, Field

from sciona.ghost.abstract import AbstractArray


class AbstractKDTree(BaseModel):
    """Abstract representation of a KD-tree carrying spatial metadata."""

    n: int = Field(..., description="Number of points in the KD-tree")
    m: int = Field(..., description="Number of dimensions of the spatial points")


def witness_build_kd_tree(
    data: AbstractArray,
    leafsize: int = 16,
) -> AbstractKDTree:
    """Builds a binary spatial partitioning tree. Output encapsulates input shape coordinates."""
    n = data.shape[0] if len(data.shape) > 0 else 0
    m = data.shape[1] if len(data.shape) > 1 else 0
    return AbstractKDTree(n=n, m=m)


def witness_query_kd_tree(
    tree: AbstractKDTree,
    query_points: AbstractArray,
    k: int = 1,
    p: float = 2.0,
    distance_upper_bound: float = float("inf"),
) -> tuple[AbstractArray, AbstractArray]:
    """Transforms query points of shape (M, D) with neighbors k into distance and index matrices of shape (M, k)."""
    out_shape: tuple[int, ...]
    if len(query_points.shape) == 1:
        out_shape = (k,) if k > 1 else ()
    elif len(query_points.shape) == 2:
        out_shape = (query_points.shape[0], k) if k > 1 else (query_points.shape[0],)
    else:
        out_shape = ()

    return (
        AbstractArray(shape=out_shape, dtype="float64"),
        AbstractArray(shape=out_shape, dtype="int64"),
    )
