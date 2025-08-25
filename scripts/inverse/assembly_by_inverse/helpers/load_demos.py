import h5py
import numpy as np
from typing import List, Dict

def load_demos_from_hdf5(hdf5_path: str) -> List[Dict[str, np.ndarray]]:
    """
    Your provided HDF5 -> flat dict converter.
    """
    demos: List[Dict[str, np.ndarray]] = []
    with h5py.File(hdf5_path, "r") as f:
        for group_name in f.keys():
            group = f[group_name]
            for demo_name in group.keys():
                demo_group = group[demo_name]

                actions = np.array(demo_group["actions"])
                obs_group = demo_group["obs"]
                obs_group_key_order = ["joint_positions", "object_pos", "object_quat", "target_pos", "ft_sensor"] # IMPORTANT: Ensure this matches your actual observation key order

                obs_arrays = []
                for k in obs_group_key_order:
                    arr = np.array(obs_group[k])
                    obs_arrays.append(arr.reshape(arr.shape[0], -1))

                observations = np.concatenate(obs_arrays, axis=-1)

                T = actions.shape[0]
                if observations.shape[0] > T:
                    observations = observations[:T]
                elif observations.shape[0] < T:
                    actions = actions[:observations.shape[0]]

                demos.append({"observations": observations, "actions": actions})
    return demos