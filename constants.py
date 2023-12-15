import torch
import sentinel1decoder
import configparser
from ast import literal_eval
import numpy as np 
import pandas as pd

config = configparser.ConfigParser()
config.read("model_setting.ini")

PATCH_DIM = literal_eval(config['TRAINER']['PATCH_DIM'])
X_RANGE = literal_eval(config['TILER']['X_RANGE'])

def load_constants_from_meta(meta, patch_dim = PATCH_DIM):
    constants = {}
    constants['wavelength'] = sentinel1decoder.constants.TX_WAVELENGTH_M
    constants['c'] = sentinel1decoder.constants.SPEED_OF_LIGHT_MPS  # Speed of light
    constants['PRI'] = meta["PRI"].unique()[0]
    constants['len_az_line'] = patch_dim[0]
    constants['len_range_line'] = patch_dim[1]
    constants['az_sample_freq'] = 1 / constants['PRI']
    RGDEC = meta["Range Decimation"].unique()[0]
    constants['range_sample_freq'] = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
    constants['range_sample_period'] = 1 / constants['range_sample_freq']
    constants['pi'] = 3.141592653589793
    constants['rank'] = meta["Rank"].unique()[0]
    suppressed_data_time = 320/(8*sentinel1decoder.constants.F_REF)

    
    constants['range_start_time'] = meta["SWST"].unique()[0] + suppressed_data_time
    # TODO: verify the correct application of RST
    # print('Before:')
    # print(constants['range_start_time'])
    # print('After:')
    # constants['range_start_time'] = meta["SWST"].unique()[0] + suppressed_data_time 
    # print(constants['range_start_time'])
    # scale_factor = constants['len_range_line']/19950
    
    constants['sample_num_along_range_line'] = torch.arange(start=0, end=constants['len_range_line'], step=1, dtype=torch.float64)
    fast_time_vec = constants['range_start_time'] + (constants['range_sample_period'] * constants['sample_num_along_range_line'])
    constants['fast_time_vec'] = fast_time_vec
    
    # REPLICA:
    TXPSF = meta["Tx Pulse Start Frequency"].unique()[0]
    TXPRR = meta["Tx Ramp Rate"].unique()[0] 
    TXPL = meta["Tx Pulse Length"].unique()[0] 
    num_tx_vals = int(TXPL*constants['range_sample_freq'])
    tx_replica_time_vals = torch.linspace(-TXPL/2, TXPL/2, steps=num_tx_vals)
    phi1 = TXPSF + TXPRR*TXPL/2
    phi2 = TXPRR/2
    tx_replica = torch.exp(2j * constants['pi'] * (phi1*tx_replica_time_vals + phi2*tx_replica_time_vals**2))
    constants['tx_replica'] = tx_replica
    return constants


# def load_constants():
#     constants = {}
#     constants['wavelength'] = torch.tensor(0.055465764662349676) 
#     constants['c'] = torch.tensor(299792458.0)  # Speed of light
    
#     constants['PRI'] = torch.tensor(0.0005345716926237736)
#     constants['len_az_line'] = 18710
#     constants['len_range_line'] = 25780
#     constants['az_sample_freq'] = 1 / constants['PRI']
#     start = -constants['az_sample_freq'] / 2
#     end = constants['az_sample_freq'] / 2
#     step = 1 / (constants['PRI'] * constants['len_az_line'])
#     constants['f_eta'] = torch.arange(start=start, end=end, step=step, dtype=torch.float32)
#     constants['range_sample_freq'] = 100092592.64 # ok
#     constants['range_sample_period'] = 1 / constants['range_sample_freq']
#     constants['pi'] = 3.141592653589793
#     constants['rank'] = 9
#     constants['range_start_time'] = 0.00011360148005720262
#     constants['sample_num_along_range_line'] = torch.arange(start=0, end=constants['len_range_line'], step=1)
#     fast_time_vec = constants['range_start_time'] + (constants['range_sample_period'] * constants['sample_num_along_range_line'])
#     constants['fast_time_vec'] = fast_time_vec

#     return constants


if __name__ == '__main__':
    print(PATCH_DIM)
    print(type(PATCH_DIM))
    raw_path = '/media/warmachine/DBDISK/SSFocus/data/Processed/Sao_Paolo/raw_s1b-s6-raw-s-vv-20210103t214313-20210103t214344-024995-02f995.pkl'

    raw = pd.read_pickle(raw_path)

    echo = raw['echo']
    aux = raw['metadata']
    const = load_constants_from_meta(meta=aux, patch_dim = PATCH_DIM)
    
    