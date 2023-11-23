import torch

def load_constants():
    constants = {}
    constants['wavelength'] = torch.tensor(0.055465764662349676) 
    constants['c'] = torch.tensor(299792458.0)  # Speed of light
    
    constants['PRI'] = torch.tensor(0.0005345716926237736)
    constants['len_az_line'] = 18710
    constants['len_range_line'] = 25780
    constants['az_sample_freq'] = 1 / constants['PRI']
    start = -constants['az_sample_freq'] / 2
    end = constants['az_sample_freq'] / 2
    step = 1 / (constants['PRI'] * constants['len_az_line'])
    constants['f_eta'] = torch.arange(start=start, end=end, step=step, dtype=torch.float32)
    constants['range_sample_freq'] = 100092592.64 # ok
    constants['range_sample_period'] = 1 / constants['range_sample_freq']
    constants['pi'] = 3.141592653589793
    constants['rank'] = 9
    constants['range_start_time'] = 0.00011360148005720262
    constants['sample_num_along_range_line'] = torch.arange(start=0, end=constants['len_range_line'], step=1)
    fast_time_vec = constants['range_start_time'] + (constants['range_sample_period'] * constants['sample_num_along_range_line'])
    constants['fast_time_vec'] = fast_time_vec

    return constants
