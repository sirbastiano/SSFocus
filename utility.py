import matplotlib.pyplot as plt  
from matplotlib.colors import LogNorm
import numpy as np
import os 
import zarr 
from matplotlib import colors
import torch

def plot_with_logscale(res, k=1):
    plt.figure(dpi=300, figsize=(12,12))
    if type(res) == torch.Tensor:
        res = np.abs(res.detach().cpu().numpy())
    else:
        res = np.abs(res)
    plt.imshow(res, norm=LogNorm(vmin=res.mean()-k*res.std(), vmax=res.mean()+k*res.std()))  # vmin should be > 0 for LogNorm
    plt.colorbar
    
def read_zarr_database():
    file_path = "/home/roberto/PythonProjects/SSFocus/Data/FOCUSED/Mini_R2F.zarr"
    # To read the root array or group
    root = read_zarr_file(file_path)
    # To read a specific array or group
    raw = read_zarr_file(file_path, "raw")
    gt = read_zarr_file(file_path, "gt")
    return raw, gt    

def read_zarr_file(file_path, array_or_group_key=None):
    """
    Read and extract data from a .zarr file.

    Parameters:
    - file_path: str, the path to the .zarr file.
    - array_or_group_key: str, optional key specifying which array or group to extract from the Zarr store.

    Returns:
    Zarr array or group, depending on what is stored in the file.
    """
    # Open Zarr file
    root = zarr.open(file_path, mode='r')

    if array_or_group_key is None:
        # Return the root group or array if no key is specified
        return root
    else:
        # Otherwise, return the specified array or group
        return root[array_or_group_key]


def get_lognorm(output_data):
    output_data_mean = np.mean(np.abs(output_data))
    output_data_std = np.std(np.abs(output_data))
    norm_output = colors.LogNorm(vmin=output_data_mean - output_data_std * 0.5 + 1e-10, 
                          vmax=output_data_mean + output_data_std * 2)
    return norm_output

def find_checkpoint(folder):
    subdir = os.path.join(folder, 'checkpoints')
    checkpoint_filepath = os.listdir(subdir)[0]
    return os.path.join(subdir, checkpoint_filepath)    
    
def main():
    print("Starting main..")
    pass


if __name__ == '__main__':
    main()