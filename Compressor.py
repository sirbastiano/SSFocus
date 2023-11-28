import torch 
from torch import nn 

import matplotlib.pyplot as plt
from constants import load_constants
import pandas as pd 

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class estimate_V_eff(nn.Module):
    def __init__(self, V_aux, patch_dim=(4096, 4096)):
        super().__init__()
        Vx, Vy, Vz = V_aux
        self.patch_dim = patch_dim
        self.V0 = torch.sqrt(torch.tensor(Vx*Vx + Vy*Vy + Vz*Vz)).to(device)

        self.a0 = nn.Parameter(torch.tensor(1.0))  # a0 parameter
        self.ar = nn.Parameter(torch.tensor(1e-5))  # ar parameter (riga)
        self.an = nn.Parameter(torch.tensor(-1e-5))  # an parameter (colonna)

    def forward(self):
        x = torch.ones(self.patch_dim, device=device) * self.V0
        # A0 addition
        x += self.a0
        # Create a range tensor for rows and columns
        r = torch.arange(self.patch_dim[1], device=x.device).view(-1, 1) * self.ar
        c = torch.arange(self.patch_dim[0], device=x.device).view(1, -1) * self.an
        # Add to x using broadcasting
        x += r
        x += c
        return x
            
class estimate_D(nn.Module):
    def __init__(self, start_idx: int, end_idx: int):
        super().__init__()
        self.constants = load_constants()
        self.wavelength = self.constants['wavelength'] 
        PRI = self.constants['PRI'] 
        len_az_line = self.constants['len_az_line'] 
        az_sample_freq = 1 / PRI
        start = -az_sample_freq / 2
        end = az_sample_freq / 2
        step = 1 / (PRI * len_az_line)
        self.f_eta = torch.arange(start=start, end=end, step=step, dtype=torch.float32)[start_idx:end_idx]
        
    def forward(self, V):
        A = (self.wavelength**2 * self.f_eta**2).to(device)
        # print(A)
        B = torch.tensor(4.0, device=device) * V*V
        # print(B)
        C = A/B
        E = 1 - C
        F = torch.sqrt(E)
        return F.T
   
class Focalizer(nn.Module):
    def __init__(self, metadata: dict):
        super().__init__()
        self.device = device
        self.constants = load_constants()
        
        self.V_metadata = self.extract_velocity(metadata) 
        # slant range vector:
        self.R0 = ((self.constants['rank'] * self.constants['PRI']) + self.constants['fast_time_vec']) * self.constants['c']/2
        self.patch_dim = (4096, 4096)
        self.start_idx, self.end_idx = 0, 4096
        
        # layers initialization:
        self.V = estimate_V_eff(self.V_metadata, self.patch_dim)
        
        
    @staticmethod
    def extract_velocity(metadata):
        ephemeris = metadata['ephemeris']
        Vx = ephemeris["X-axis velocity ECEF"][0].item()
        Vy = ephemeris["Y-axis velocity ECEF"][0].item()
        Vz = ephemeris["Z-axis velocity ECEF"][0].item()
        return (Vx,Vy,Vz) 

    def _compute_filter_rcmc(self):
        """ Compute the RCMC shift filter. """
        self.D = estimate_D(start_idx=0, end_idx=4096)(self.V())
        self.R0 = self.R0[self.start_idx:self.end_idx].to(device)
        
        # self.RO is slant range vec
        rcmc_shift = self.R0 * ( (1/self.D) - 1)
        range_freq_vals = torch.linspace(-self.constants['range_sample_freq']/2, self.constants['range_sample_freq']/2, steps=self.constants['len_range_line'])[self.start_idx:self.end_idx]
        self.rcmc_filter = torch.exp(4j * self.constants['pi'] * range_freq_vals.to(self.device) * rcmc_shift / self.constants['c'])
        return self.rcmc_filter

    def _compute_azimuth_filter(self):
        """ Compute the Azimuth filter. """
        self.azimuth_filter = torch.exp(4j * self.constants['pi'] * self.R0 * self.D / self.constants['wavelength']).to(self.device)
        return self.azimuth_filter

    def forward(self, X):
        # TODO: this operation should be implemented for the batch and not for a single element 2-C
        X = X * self._compute_filter_rcmc().unsqueeze(0).unsqueeze(0)
        X = torch.fft.ifftshift(torch.fft.ifft(X, dim=-1), dim=-1) # Convert to Range-Doppler
        X = X * self._compute_azimuth_filter().unsqueeze(0).unsqueeze(0)
        X = torch.fft.ifft(X, dim=-2)
        return X   


if __name__ == '__main__':

        
    def test_model():
        x = torch.load('/home/roberto/PythonProjects/SSFocus/Data/4096_test_fft2D.pt').to(device)
        aux = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_pkt_8_metadata.pkl')
        eph = pd.read_pickle('/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a_ephemeris.pkl')
        model = Focalizer(metadata={'aux':aux, 'ephemeris':eph})
        print('Model parameters:', list(model.parameters()))
    
    test_model()