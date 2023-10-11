# import numpy as np
# import argparse
# import torch
# import os
# import pickle
# import pandas as pd
# import torch
# from pathlib import Path 
# import copy 

# # check ram usage:
# import psutil

# def printmemory():
#     print(f'RAM memory usage: {psutil.virtual_memory().percent}%')
#     return

# # global variables:
# global device, slant_range_vec, D, c, len_range_line, range_sample_freq, wavelength

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
# # divide the data into 10 parts:
# def get_partition(data_path: str = 'path/to/*.npy', ephem_path: str = 'path/to/ephem_file', meta_path: str = 'path/to/metafile', num_chunks: int = 5, idx_chunk: int = 0):
#     """
#     Get a partition of the data from a numpy file and corresponding metadata and ephemeris files.

#     Args:
#         data_path (str): Path to the numpy data file.
#         ephem_path (str): Path to the ephemeris file.
#         meta_path (str): Path to the metadata file.
#         num_chunks (int): Number of chunks to divide the data into.
#         idx_chunk (int): Index of the chunk to load.

#     Returns:
#         tuple: A tuple containing the partition of the data, metadata, and ephemeris.
#     """
#     global len_az_line, len_range_line
#     data = np.load(data_path)
#     # Image sizes
#     len_range_line = data[1]
#     len_az_line = data[0]
#     start = int(idx_chunk * data.shape[0] / num_chunks)
#     end = int((idx_chunk + 1) * data.shape[0] / num_chunks)
#     partition = data[start:end, :]
    
#     # copy partition deepcopy
#     copy_partition = copy.deepcopy(partition)
#     del partition, data
    
#     meta = pd.read_pickle(meta_path)[start:end]
#     ephemeris = pd.read_pickle(ephem_path)[start:end]
#     print('- Data loaded successfully!')
#     return copy_partition, meta, ephemeris

# def picklesavefile(path, datafile):
#     with open(path, 'wb') as f:
#         pickle.dump(datafile, f)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='SAR Processor')
#     parser.add_argument('--data', type=str, default='radar_data.npy', help='path to the radar data')
#     parser.add_argument('--meta', type=str, default='/path/to/ephemeris.pkl', help='Path to the ephemeris file')
#     parser.add_argument('--ephemeris', type=str, default='radar_data.npy', help='path to the radar data')
#     parser.add_argument('--output', type=str, default='outputdir', help='path to the focused radar data')
#     parser.add_argument('--num_chunks', type=int, default=15, help='Number of chunks to parse the SAR data')
#     parser.add_argument('--idx_chunk', type=int, default=0, help='Index of the chunk to parse the SAR data')
    
#     print('\n\n***   Starting SAR Raw Patcher   ***')
#     args = parser.parse_args()
#     # Load data:
#     name = Path(args.data).stem
#     idx = args.idx_chunk
#     radar_data, meta, ephemeris = get_partition(data_path=args.data, ephem_path=args.ephemeris, meta_path=args.meta, num_chunks = args.num_chunks, idx_chunk=idx)
#     output_dict = {'echo': radar_data, 'meta': meta, 'ephemeris': ephemeris, 'LOsize':[len_az_line, len_range_line]}
#     picklesavefile(os.path.join(args.output, f'raw_{idx}-{args.num_chunks}_{name}.pkl'), output_dict)




import os, sys
import copy
import numpy as np
import pandas as pd
from pathlib import Path
import numpy as np
import pickle


def main(idx_chunk, product_idx):
    folderpath = '/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/'
    bs_outfolder = '/home/roberto/PythonProjects/SSFocus/Data/FOCUSED/'

    npyfiles = [x for x in Path(folderpath).iterdir() if x.is_file() and x.suffix == '.npy']

    def picklesavefile(path, datafile):
        with open(path, 'wb') as f:
            pickle.dump(datafile, f)


    num_chunks = 10
    data = np.load(npyfiles[product_idx])
    name = Path(npyfiles[product_idx]).stem
    start = int(idx_chunk * data.shape[0] / num_chunks)
    end = int((idx_chunk + 1) * data.shape[0] / num_chunks)
    print('Start: {}, End: {}'.format(start, end))
    partition = data[start:end, :]
    outfolder = bs_outfolder + name + '/'
    picklesavefile(outfolder+f'{name}_datafile_{idx_chunk}.pkl', partition)
    print('Saved datafile_{}.pkl'.format(idx_chunk))
    
    
    
if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))