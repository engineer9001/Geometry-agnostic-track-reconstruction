import numpy as np
from typing import Iterable, Mapping, Optional, Sequence, Tuple, List
import os
import tempfile
import unittest

try:
    import h5py
except ImportError:
    h5py = None


def build_channel_index(hit_ids: Iterable[str], sort: bool = True) -> Mapping[str, int]:
    """Create a stable channel-to-index map for a fixed dense representation."""
    if sort:
        unique_ids = sorted(set(hit_ids))
    else:
        unique_ids = list(dict.fromkeys(hit_ids))
    return {channel_id: idx for idx, channel_id in enumerate(unique_ids)}


def sparse_hits_to_dense_matrix(
    hit_ids: Sequence[str],
    features: np.ndarray,
    channel_index: Mapping[str, int],
    n_channels: Optional[int] = None,
    padding_value: float = 0.0,
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Build a temporarily padded dense tensor from sparse hit data."""
    if dtype is None:
        dtype = features.dtype if isinstance(features, np.ndarray) else np.float32

    if n_channels is None:
        n_channels = max(channel_index.values()) + 1

    n_features = features.shape[1]
    dense = np.full((n_channels, n_features), padding_value, dtype=dtype)

    for idx, hit_id in enumerate(hit_ids):
        if hit_id not in channel_index:
            raise KeyError(f"Hit channel '{hit_id}' not found in channel_index.")
        dense[channel_index[hit_id]] = features[idx]

    return dense


def flatten_dense_matrix(dense: np.ndarray, order: str = "C") -> np.ndarray:
    """Flatten a dense matrix into a 1D ML-ready feature vector."""
    return dense.ravel(order=order)


def build_sparse_track_from_hdf5_group(
    track_group: "h5py.Group",
    channel_index: Mapping[str, int],
    feature_names: Sequence[str] = ("t_diff", "edep"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts hits from HDF5 and maps channel strings directly to integer IDs.
    
    Note: Track-level target vector components (true_mom_x, y, z) are stored as 
    group-level attributes and are read directly inside the PyTorch Dataset class.
    """
    hits = track_group["hits"]
    hit_ids = [hid.decode("utf-8") if isinstance(hid, bytes) else str(hid) for hid in hits["hit_id"]]

    # Stack feature columns directly into a (n_hits, n_features) array.
    # np.column_stack avoids the intermediate list-of-arrays from np.stack.
    feature_arrays = [np.asarray(hits[name], dtype=np.float32) for name in feature_names]
    features = np.column_stack(feature_arrays) if len(feature_arrays) > 1 else feature_arrays[0][:, None]

    # Vectorized channel lookup: np.vectorize dispatches the dict.__getitem__
    # call as a ufunc, avoiding the explicit Python for-loop over hit_ids.
    indices = np.vectorize(channel_index.__getitem__)(hit_ids).astype(np.int64)

    return features, indices


def build_dense_track_from_hdf5_group(
    track_group: "h5py.Group",
    channel_index: Mapping[str, int],
    feature_names: Sequence[str] = ("t_diff", "edep"),
    padding_value: float = 0.0,
    n_channels: Optional[int] = None,
    flatten: bool = False,
    order: str = "C",
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn a compact per-track HDF5 hit group into a padded dense track object."""
    hits = track_group["hits"]
    hit_ids = [hid.decode("utf-8") if isinstance(hid, bytes) else str(hid) for hid in hits["hit_id"]]

    feature_arrays = [np.asarray(hits[name], dtype=np.float32) for name in feature_names]
    features = np.stack(feature_arrays, axis=-1)

    dense = sparse_hits_to_dense_matrix(
        hit_ids=hit_ids,
        features=features,
        channel_index=channel_index,
        n_channels=n_channels,
        padding_value=padding_value,
    )

    mask = np.zeros(dense.shape[0], dtype=bool)
    for hit_id in hit_ids:
        mask[channel_index[hit_id]] = True

    if flatten:
        dense = flatten_dense_matrix(dense, order=order)

    return dense, mask


def infer_channel_index_from_hdf5(
    h5_path: str,
    group_root: str = "tracks",
    sort: bool = True,
    max_tracks: Optional[int] = None,
) -> Mapping[str, int]:
    """Scan an HDF5 file and build a stable channel index from all observed hit IDs."""
    unique_channels: List[str] = []
    seen: set = set()

    with h5py.File(h5_path, "r") as f:
        group_root_obj = f[group_root]
        for i, track_name in enumerate(group_root_obj):
            if max_tracks is not None and i >= max_tracks:
                break
            hits = group_root_obj[track_name]["hits"]
            for raw_hit_id in hits["hit_id"]:
                hit_id = raw_hit_id.decode("utf-8") if isinstance(raw_hit_id, bytes) else str(raw_hit_id)
                if hit_id not in seen:
                    seen.add(hit_id)
                    unique_channels.append(hit_id)

    if sort:
        unique_channels = sorted(unique_channels)
    return {channel_id: idx for idx, channel_id in enumerate(unique_channels)}


if __name__ == "__main__":
    import argparse

    class TestNEDT(unittest.TestCase):
        def test_build_channel_index_sorting_preserves_order(self):
            channel_ids = ["p1", "p3", "p2", "p3", "p1"]
            sorted_index = build_channel_index(channel_ids, sort=True)
            self.assertEqual(sorted_index, {"p1": 0, "p2": 1, "p3": 2})
            order_index = build_channel_index(channel_ids, sort=False)
            self.assertEqual(order_index, {"p1": 0, "p3": 1, "p2": 2})

        def test_sparse_hits_to_dense_matrix_basic_padding(self):
            channel_index = {"a": 0, "b": 1, "c": 2}
            hit_ids = ["a", "c"]
            features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            dense = sparse_hits_to_dense_matrix(hit_ids, features, channel_index, padding_value=-1.0)
            expected = np.array([
                [1.0, 2.0],
                [-1.0, -1.0],
                [3.0, 4.0],
            ], dtype=np.float32)
            np.testing.assert_array_equal(dense, expected)

        def test_sparse_hits_to_dense_matrix_missing_channel_raises(self):
            channel_index = {"a": 0, "b": 1}
            hit_ids = ["a", "c"]
            features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
            with self.assertRaises(KeyError):
                sparse_hits_to_dense_matrix(hit_ids, features, channel_index)

    parser = argparse.ArgumentParser(description="nedt.py layout utility wrapper.")
    parser.add_argument("--run-tests", action="store_true")
    args = parser.parse_args()

    if args.run_tests:
        unittest.main(argv=[""], exit=True, verbosity=2)