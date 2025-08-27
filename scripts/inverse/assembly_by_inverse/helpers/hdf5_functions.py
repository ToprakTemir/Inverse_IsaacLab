import h5py
import numpy as np
from typing import Dict, Any, List, Tuple, Union

ArrayLike = Union[np.ndarray, float, int]

def _to_numpy(ds: h5py.Dataset) -> np.ndarray:
    # Reads any HDF5 dataset into a np.ndarray (including scalars)
    arr = ds[()]
    return np.array(arr)

def h5_to_nested_dict(h5obj: Union[h5py.Group, h5py.Dataset]) -> Any:
    """Recursively convert an HDF5 Group/Dataset into nested Python dicts / numpy arrays."""
    if isinstance(h5obj, h5py.Dataset):
        return _to_numpy(h5obj)
    elif isinstance(h5obj, h5py.Group):
        out = {}
        for k in h5obj.keys():
            out[k] = h5_to_nested_dict(h5obj[k])
        return out
    else:
        raise TypeError(f"Unsupported HDF5 type: {type(h5obj)}")

def print_schema(nested: Any, prefix: str = "") -> None:
    """Print flattened keys with shapes/dtypes to understand structure."""
    if isinstance(nested, dict):
        for k, v in nested.items():
            new_prefix = f"{prefix}/{k}" if prefix else k
            print_schema(v, new_prefix)
    else:
        # leaf: numpy array (or scalar)
        arr = np.array(nested)
        shape = arr.shape
        dtype = arr.dtype
        print(f"{prefix} -> shape={shape}, dtype={dtype}")

def load_isaaclab_demos(path: str, data_group: str = "data") -> List[Dict[str, Any]]:
    """Loads all demos under /data/demo_* (or whatever is there) into a list of nested dicts."""
    episodes = []
    with h5py.File(path, "r") as f:
        assert data_group in f, f"'{data_group}' group not found. Top-level: {list(f.keys())}"
        grp = f[data_group]
        # sort ensures deterministic order demo_0, demo_1, ...
        for name in sorted(grp.keys()):
            demo = grp[name]
            episodes.append(h5_to_nested_dict(demo))
    return episodes


# --- Example usage ---
if __name__ == "__main__":
    path = "datasets/disassembly_15.hdf5"
    episodes = load_isaaclab_demos(path, data_group="data")

    # Inspect schema of the first demo
    print("Schema for demo_0:")
    print_schema(episodes[0])
