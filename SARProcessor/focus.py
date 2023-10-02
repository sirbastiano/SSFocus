import numpy as np
import argparse
import torch
import pickle
import sentinel1decoder
import pandas as pd
from scipy.interpolate import interp1d
import math
import torch
from pathlib import Path 
import copy 
import gc
# check ram usage:
import psutil

def printmemory():
    print(f'RAM memory usage: {psutil.virtual_memory().percent}%')
    return

# global variables:
global device, slant_range_vec, D, c, len_range_line, range_sample_freq, wavelength

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fft2D(radar_data: np.array, backend: str = 'numpy', print_status: bool = True):
    # TODO: Test this function
    """
    Perform 2D FFT on a radar data array in range and azimuth dimensions.

    Args:
        radar_data (np.array): 2D numpy array of radar data.
        backend (str, optional): Backend to use for FFT. Defaults to 'numpy'.

    Returns:
        np.array: 2D numpy array of radar data after 2D FFT.
    """
    
    if backend == 'numpy':
        # FFT each range line
        radar_data = np.fft.fft(radar_data, axis=1)
        # FFT each azimuth line
        radar_data = np.fft.fftshift(np.fft.fft(radar_data, axis=0), axes=0)
    elif backend == 'torch':
        # Convert radar_data to a PyTorch tensor and move to device
        radar_data_tensor = torch.tensor(radar_data, dtype=torch.complex64, device=device)
        # FFT each range line
        radar_data_tensor = torch.fft.fft(radar_data_tensor, dim=1)
        # FFT each azimuth line
        radar_data_tensor = torch.fft.fftshift(torch.fft.fft(radar_data_tensor, dim=0), dim=0)
        return radar_data_tensor
    else:
        raise ValueError('Backend not supported.')
    if print_status:
        print('- FFT performed successfully!')
    return radar_data

def get_range_filter(metadata: pd.DataFrame, ephemeris: pd.DataFrame) -> np.ndarray:
    """
    Computes a range filter for radar data, specifically tailored to Sentinel-1 radar parameters.

    Parameters:
        radar_data_shape (tuple): Shape of the radar data array, typically (Azimuth, Range).
        metadata (pd.DataFrame): Dataframe containing metadata related to radar pulse parameters.
        ephemeris (pd.DataFrame): Dataframe containing ephemeris data with spacecraft's position and velocity.

    Returns:
        np.ndarray: A complex-valued 1D array representing the computed range filter.

    Notes:
        1. This function assumes that the Sentinel-1 specific constants are available through the 'sentinel1decoder.constants' module.
        2. The function makes use of the scipy `interp1d` function to interpolate spacecraft velocities.
        3. It assumes that the metadata DataFrame has specific columns like 'Range Decimation', 'PRI', 'Rank', and 'SWST'.

    Example:
        radar_data_shape = (100, 200)
        metadata = pd.DataFrame({"Range Decimation": [...], "PRI": [...], ...})
        ephemeris = pd.DataFrame({"POD Solution Data Timestamp": [...], "X-axis velocity ECEF": [...], ...})
        range_filter = get_range_filter(radar_data_shape, metadata, ephemeris)
    """
    global device, slant_range_vec, D, c, len_range_line, range_sample_freq, wavelength
    
    # Tx pulse parameters
    c = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS
    RGDEC = metadata["Range Decimation"].unique()[0]
    PRI = metadata["PRI"].unique()[0]
    rank = metadata["Rank"].unique()[0]
    suppressed_data_time = 320/(8*sentinel1decoder.constants.F_REF)
    range_start_time = metadata["SWST"].unique()[0] + suppressed_data_time
    wavelength = sentinel1decoder.constants.TX_WAVELENGTH_M

    # Sample rates
    range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    range_sample_period = 1/range_sample_freq
    az_sample_freq = 1 / PRI
    az_sample_period = PRI

    # Fast time vector - defines the time axis along the fast time direction
    sample_num_along_range_line = np.arange(0, len_range_line, 1)
    fast_time_vec = range_start_time + (range_sample_period * sample_num_along_range_line)

    # Slant range vector - defines R0, the range of closest approach, for each range cell
    slant_range_vec = ((rank * PRI) + fast_time_vec) * c/2
        
    # Axes - defines the frequency axes in each direction after FFT
    SWL = len_range_line/range_sample_freq
    az_freq_vals = np.arange(-az_sample_freq/2, az_sample_freq/2, 1/(PRI*len_az_line))
    range_freq_vals = np.arange(-range_sample_freq/2, range_sample_freq/2, 1/SWL)
    
    # Spacecraft velocity - numerical calculation of the effective spacecraft velocity
    ecef_vels = ephemeris.apply(lambda x: math.sqrt(x["X-axis velocity ECEF"]**2 + x["Y-axis velocity ECEF"]**2 +x["Z-axis velocity ECEF"]**2), axis=1)
    velocity_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ecef_vels.unique(), fill_value="extrapolate")
    x_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ephemeris["X-axis position ECEF"].unique(), fill_value="extrapolate")
    y_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ephemeris["Y-axis position ECEF"].unique(), fill_value="extrapolate")
    z_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ephemeris["Z-axis position ECEF"].unique(), fill_value="extrapolate")
    space_velocities = metadata.apply(lambda x: velocity_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

    x_positions = metadata.apply(lambda x: x_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
    y_positions = metadata.apply(lambda x: y_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)
    z_positions = metadata.apply(lambda x: z_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_numpy().astype(float)

    position_array = np.transpose(np.vstack((x_positions, y_positions, z_positions)))

    a = sentinel1decoder.constants.WGS84_SEMI_MAJOR_AXIS_M
    b = sentinel1decoder.constants.WGS84_SEMI_MINOR_AXIS_M
    H = np.linalg.norm(position_array, axis=1)
    W = np.divide(space_velocities, H)
    lat = np.arctan(np.divide(position_array[:, 2], position_array[:, 0]))
    local_earth_rad = np.sqrt(
        np.divide(
            (np.square(a**2 * np.cos(lat)) + np.square(b**2 * np.sin(lat))),
            (np.square(a * np.cos(lat)) + np.square(b * np.sin(lat)))
        )
    )
    cos_beta = (np.divide(np.square(local_earth_rad) + np.square(H) - np.square(slant_range_vec[:, np.newaxis]) , 2 * local_earth_rad * H))
    ground_velocities = local_earth_rad * W * cos_beta

    effective_velocities = np.sqrt(space_velocities * ground_velocities)

    D = np.sqrt(
        1 - np.divide(
            wavelength**2 * np.square(az_freq_vals),
            4 * np.square(effective_velocities)
        )
    ).T
    
    # Create replica pulse
    TXPSF = metadata["Tx Pulse Start Frequency"].unique()[0]
    TXPRR = metadata["Tx Ramp Rate"].unique()[0]
    TXPL = metadata["Tx Pulse Length"].unique()[0]
    num_tx_vals = int(TXPL*range_sample_freq)
    tx_replica_time_vals = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals)
    phi1 = TXPSF + TXPRR*TXPL/2
    phi2 = TXPRR/2
    tx_replica = np.exp(2j * np.pi * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2))

    # Create range filter from replica pulse
    range_filter = np.zeros(len_range_line, dtype=complex)
    index_start = np.ceil((len_range_line-num_tx_vals)/2)-1
    index_end = num_tx_vals+np.ceil((len_range_line-num_tx_vals)/2)-2
    range_filter[int(index_start):int(index_end+1)] = tx_replica
    range_filter = np.conjugate(np.fft.fft(range_filter))
    print('- Range compression performed successfully!')    
    return range_filter

def get_RDMC(metadata: pd.DataFrame):
    """
    Calculate and return the RCMC filter for a given radar dataset.

    Args:
        metadata (pd.DataFrame): Pandas DataFrame containing metadata for the radar dataset.

    Returns:
        np.array: 1D numpy array representing the RCMC filter.
    """
    global device, slant_range_vec, D, c, len_range_line, range_sample_freq, wavelength
    
    RGDEC = metadata["Range Decimation"].unique()[0]
    range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    # Create RCMC filter
    range_freq_vals = np.linspace(-range_sample_freq/2, range_sample_freq/2, num=len_range_line)
    rcmc_shift = slant_range_vec[0] * (np.divide(1, D) - 1)
    rcmc_filter = np.exp(4j * np.pi * range_freq_vals * rcmc_shift / c)
    return rcmc_filter
    
def get_azimuth_filter():
    global slant_range_vec, D, wavelength
    # Create filter
    az_filter = np.exp(4j * np.pi * slant_range_vec * D / wavelength)
    return az_filter

def multiply(a, b, backend: str ='numpy'):
    """
    Multiply two complex-valued arrays.

    Args:
        a (np.array): First complex-valued array.
        b (np.array): Second complex-valued array.
        backend (str, optional): Backend to use for multiplication. Defaults to 'numpy'.

    Returns:
        np.array: Complex-valued array after multiplication.
    """
    if backend == 'numpy':
        return np.multiply(a, b)
    elif backend == 'torch':
        # if b not a tensor convet it to a tensor_
        if not torch.is_tensor(b):
            b = torch.tensor(b, dtype=torch.complex64, device=device)
        return torch.multiply(a, b)
    else:
        raise ValueError('Backend not supported.')
    
# divide the data into 10 parts:
def get_partition(data_path: str = 'path/to/*.npy', ephem_path: str = 'path/to/ephem_file', meta_path: str = 'path/to/metafile', num_chunks: int = 5, idx_chunk: int = 0):
    """
    Get a partition of the data from a numpy file and corresponding metadata and ephemeris files.

    Args:
        data_path (str): Path to the numpy data file.
        ephem_path (str): Path to the ephemeris file.
        meta_path (str): Path to the metadata file.
        num_chunks (int): Number of chunks to divide the data into.
        idx_chunk (int): Index of the chunk to load.

    Returns:
        tuple: A tuple containing the partition of the data, metadata, and ephemeris.
    """
    global len_az_line, len_range_line
    data = np.load(data_path)
    # Image sizes
    len_range_line = data[1]
    len_az_line = data[0]
    start = int(idx_chunk * data.shape[0] / num_chunks)
    end = int((idx_chunk + 1) * data.shape[0] / num_chunks)
    partition = data[start:end, :]
    
    # copy partition deepcopy
    copy_partition = copy.deepcopy(partition)
    del partition, data
    
    meta = pd.read_pickle(meta_path)[start:end]
    ephemeris = pd.read_pickle(ephem_path)[start:end]
    print('- Data loaded successfully!')
    return copy_partition, meta, ephemeris

def picklesavefile(path, datafile):
    with open(path, 'wb') as f:
        pickle.dump(datafile, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAR Processor')
    parser.add_argument('--data', type=str, default='radar_data.npy', help='path to the radar data')
    parser.add_argument('--meta', type=str, default='/path/to/ephemeris.pkl', help='Path to the ephemeris file')
    parser.add_argument('--ephemeris', type=str, default='radar_data.npy', help='path to the radar data')
    parser.add_argument('--output', type=str, default='outputdir', help='path to the focused radar data')
    parser.add_argument('--backend', type=str, default='numpy', help='backend used to process data')
    parser.add_argument('--num_chunks', type=int, default=15, help='Number of chunks to parse the SAR data')
    parser.add_argument('--idx_chunk', type=int, default=0, help='Index of the chunk to parse the SAR data')
    
    print('\n\n***   Starting SAR Processor   ***')
    args = parser.parse_args()
    # Load data:
    name = Path(args.data).stem
    idx = args.idx_chunk
    print(f'Processing chunk {idx+1}/{args.num_chunks}')
    printmemory()
    radar_data, meta, ephemeris = get_partition(data_path=args.data, ephem_path=args.ephemeris, meta_path=args.meta, num_chunks = args.num_chunks, idx_chunk=idx)
    printmemory()
    # Processing 2dfft:
    radar_data = fft2D(radar_data, backend=args.backend)
    # range compression:
    printmemory()
    radar_data = multiply(radar_data, get_range_filter(meta, ephemeris), backend=args.backend)
    
    # applt RCMC filter:
    rcmc_filter = get_RDMC(meta)
    printmemory()
    radar_data = multiply(radar_data, rcmc_filter, backend=args.backend)
    print('- RCMC filter applied successfully!')
    printmemory()
    # Azimuth compression:
    radar_data = multiply(radar_data, get_azimuth_filter(), backend=args.backend)
    print('- Azimuth compression performed successfully!')
    printmemory()
    
    # Save focused radar data:
    picklesavefile(args.output+f'/cnk_{idx}_{name}.npy', radar_data)
    print('- Focused radar data saved successfully!')