import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def get_track_group_names(h5_path: str, group_root: str = "tracks") -> List[str]:
    """Return all track group names under the specified group root."""
    with h5py.File(h5_path, "r") as f:
        if group_root not in f:
            raise KeyError(f"Group root '{group_root}' not found in {h5_path}")
        return list(f[group_root].keys())


def split_names(
    names: Sequence[str],
    train_fraction: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
) -> Tuple[List[str], List[str]]:
    """Split a list of names into training and validation subsets."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    names = list(names)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(names)

    n_train = int(np.floor(len(names) * train_fraction))
    train_names = names[:n_train]
    val_names = names[n_train:]
    return train_names, val_names


def copy_track_groups(
    src_h5_path: str,
    dst_h5_path: str,
    track_names: Iterable[str],
    group_root: str = "tracks",
) -> None:
    """Copy a subset of track groups from one HDF5 file into another.
    
    Using src.copy() naturally guarantees that your 3D target components 
    (true_mom_x, true_mom_y, true_mom_z) are preserved as metadata attributes.
    """
    with h5py.File(src_h5_path, "r") as src, h5py.File(dst_h5_path, "w") as dst:
        if group_root in src:
            dest_root = dst.require_group(group_root)
        else:
            raise KeyError(f"Group root '{group_root}' not found in source HDF5 file")

        # Copy top-level file attributes if any
        for key, value in src.attrs.items():
            dst.attrs[key] = value

        # Copy group-level attributes from the source tracks/ group
        # (has_calo, hit_feature_cols, track_scalar_attrs, has_momentum, etc.)
        # These are written by track_aggregation.py and must be preserved so
        # that preprocess_to_flat.py knows what data is available.
        for key, value in src[group_root].attrs.items():
            dest_root.attrs[key] = value

        for name in track_names:
            src_path = f"{group_root}/{name}"
            if src_path not in src:
                raise KeyError(f"Track group '{name}' not found under '{group_root}'")
            src.copy(src_path, dest_root, name=name)


def split_h5_file(
    src_h5_path: str,
    train_h5_path: str,
    val_h5_path: str,
    train_fraction: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
    group_root: str = "tracks",
) -> Tuple[int, int]:
    """Split track groups from a single HDF5 file into train and validation files."""
    track_names = get_track_group_names(src_h5_path, group_root=group_root)
    train_names, val_names = split_names(track_names, train_fraction, seed, shuffle)

    copy_track_groups(src_h5_path, train_h5_path, train_names, group_root=group_root)
    copy_track_groups(src_h5_path, val_h5_path, val_names, group_root=group_root)

    return len(train_names), len(val_names)


def list_source_h5_files(
    source_dir: str,
    exclude_suffixes: Sequence[str] = ("_train.h5", "_val.h5"),
) -> List[str]:
    """List HDF5 files in a directory, excluding already-split files."""
    dir_path = Path(source_dir)
    if not dir_path.is_dir():
        raise ValueError(f"Source path '{source_dir}' is not a directory.")

    source_files = []
    for h5_path in sorted(dir_path.glob("*.h5")):
        if any(h5_path.name.endswith(suffix) for suffix in exclude_suffixes):
            continue
        source_files.append(str(h5_path))

    return source_files


def split_h5_directory(
    source_dir: str,
    train_fraction: float = 0.8,
    seed: int = 42,
    shuffle: bool = True,
    group_root: str = "tracks",
) -> List[Tuple[str, int, int, str, str]]:
    """Split every eligible HDF5 file in a directory into train/validation outputs."""
    source_files = list_source_h5_files(source_dir)
    if not source_files:
        raise ValueError(f"No eligible .h5 files found in directory '{source_dir}'.")

    results = []
    for src in source_files:
        src_path = Path(src)
        train_h5_path = src_path.with_name(src_path.stem + "_train.h5")
        val_h5_path = src_path.with_name(src_path.stem + "_val.h5")
        n_train, n_val = split_h5_file(
            src,
            str(train_h5_path),
            str(val_h5_path),
            train_fraction=train_fraction,
            seed=seed,
            shuffle=shuffle,
            group_root=group_root,
        )
        results.append((src, n_train, n_val, str(train_h5_path), str(val_h5_path)))

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a track HDF5 file or directory of HDF5 files into train/validation outputs."
    )
    parser.add_argument(
        "source",
        help="Source HDF5 file or directory containing .h5 files to split.",
    )
    parser.add_argument(
        "--train-out",
        default=None,
        help="Output HDF5 file for training tracks. Defaults to source_train.h5.",
    )
    parser.add_argument(
        "--val-out",
        default=None,
        help="Output HDF5 file for validation tracks. Defaults to source_val.h5.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of tracks to use for training (default: 0.8).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable shuffling before splitting.",
    )
    parser.add_argument(
        "--group-root",
        default="tracks",
        help="HDF5 group root containing the track groups (default: tracks).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)

    if source_path.is_dir():
        if args.train_out or args.val_out:
            raise ValueError("--train-out and --val-out cannot be used when source is a directory.")

        results = split_h5_directory(
            str(source_path),
            train_fraction=args.train_fraction,
            seed=args.seed,
            shuffle=not args.no_shuffle,
            group_root=args.group_root,
        )

        total_train = sum(r[1] for r in results)
        total_val = sum(r[2] for r in results)
        print(f"Processed {len(results)} source .h5 files from {source_path}:")
        for src, n_train, n_val, train_out, val_out in results:
            print(f"  {Path(src).name}: {n_train} train, {n_val} val -> {Path(train_out).name}, {Path(val_out).name}")
        print(f"Total: {total_train} train tracks, {total_val} val tracks")
        return

    if not source_path.is_file():
        raise ValueError(f"Source path '{source_path}' is not a file or directory.")

    train_path = Path(args.train_out or source_path.with_name(source_path.stem + "_train.h5"))
    val_path = Path(args.val_out or source_path.with_name(source_path.stem + "_val.h5"))

    n_train, n_val = split_h5_file(
        str(source_path),
        str(train_path),
        str(val_path),
        train_fraction=args.train_fraction,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        group_root=args.group_root,
    )

    total = n_train + n_val
    print(f"Split {total} tracks from {source_path.name} into:")
    print(f"  {n_train} train tracks -> {train_path}")
    print(f"  {n_val} val tracks   -> {val_path}")


if __name__ == "__main__":
    main()