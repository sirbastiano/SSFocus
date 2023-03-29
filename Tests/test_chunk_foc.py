import matplotlib.pyplot as plt
import sentinel1decoder
import logging
import pandas as pd
from pathlib import Path

from SARProcessor.focus import RD, get_chunks

inputfile = '/home/roberto/PythonProjects/SSFocus/Data/RAW/IW/ROME/S1A_IW_RAW__0SDV_20201203T051956_20201203T052028_035517_042709_AB3D.SAFE/s1a-iw-raw-s-vh-20201203t051956-20201203t052028-035517-042709.dat'
filename = Path(inputfile).stem
decoder = sentinel1decoder.Level0Decoder(inputfile, log_level=logging.WARNING)
raw_df = decoder.decode_metadata()
ephemeris = sentinel1decoder.utilities.read_subcommed_data(raw_df)
print(raw_df['Swath Number'].unique())
print('filename: ', filename)

try:
    chunks = get_chunks(raw_df)
except:
    print(f"The chunking failed for {inputfile}")
    
for idx, chunk in enumerate(chunks):
    focuser = RD(decoder=decoder, raw=chunk, ephemeris=ephemeris)
    img_focused = focuser.process()
    pd.to_pickle(img_focused, f"/home/roberto/PythonProjects/SSFocus/Data/RAW/IW/ROME/Processed/{idx}_chunk_{filename}.pkl")