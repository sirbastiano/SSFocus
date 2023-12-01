import torch
import torch.nn as nn
import pytorch_lightning as pl
from scipy.interpolate import interp1d

import torch.nn.functional as F
from torch.utils.data import TensorDataset

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np  
import zarr

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

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")

torch.set_float32_matmul_precision('medium')  # For performance


def fix_tensor_assignment(tensor):
    return torch.nn.Parameter(tensor.float()).to(tensor.device)



# 1. Define the Simple AI Model
##############################################################################
class ComplexCNN(nn.Module):
    def __init__(self, n_convolutions=2):
        super(ComplexCNN, self).__init__()

        # Convolution with stride equal to half filter dimension
        self.kernel_size = 7
        self.stride = self.kernel_size // 2
        
        # n_convolutions
        self.n_convolutions = n_convolutions
        
        # Create common conv function (conv + relu + batch norm)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), bias=False) 
            for _ in range(self.n_convolutions)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(2)
            for _ in range(self.n_convolutions)
        ])

        # Add a decoder to upsample to original size
        # Here we use a transposed convolution
        self.decoder = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), bias=False)

    def common_conv(self, x, conv, batch_norm):
        x = conv(x)
        x = F.relu(x)
        x = batch_norm(x)
        return x

    def forward(self, x):
        # Separate real and imaginary parts
        x_real = x.real
        x_imag = x.imag

        # Stack them into a new tensor
        x_stacked = torch.stack((x_real, x_imag), dim=1)

        # Apply n_convolutions using common_conv
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x_stacked = self.common_conv(x_stacked, conv, batch_norm)

        # Decode (upsample) to original size
        x_stacked = self.decoder(x_stacked)

        # Reconstruct the complex tensor
        out_real = x_stacked[:, 0]
        out_imag = x_stacked[:, 1]

        return torch.complex(out_real, out_imag)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        
        # Convolution with stride equal to half filter dimension
        self.kernel_size = 7
        self.stride = 1
        
        ch = [in_channels,4,8,16,out_channels]
        
        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, ch[1], kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), groups=in_channels)
        self.enc_conv2 = nn.Conv2d(ch[1], ch[2], kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), groups=ch[1])
        self.enc_conv3 = nn.Conv2d(ch[2], ch[3], kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), groups=ch[2])
        self.enc_conv4 = nn.Conv2d(ch[3], ch[3], kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), groups=ch[3])
        
        # Decoder
        self.dec_conv1 = nn.Conv2d(ch[3], ch[2], kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), groups=ch[2])
        self.dec_conv2 = nn.Conv2d(ch[2], ch[1], kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), groups=ch[1])
        self.dec_conv3 = nn.Conv2d(ch[1], out_channels, kernel_size=(self.kernel_size, 1), stride=(self.stride, 1), padding=(1, 0), groups=out_channels)
        
        # Pooling and Upsampling
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1))
        self.upsample = nn.Upsample(scale_factor=(2,1), mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoding
        
        # Separate real and imaginary parts
        x_real = x.real
        x_imag = x.imag

        # Stack them into a new tensor
        x = torch.stack((x_real, x_imag), dim=1)
        
        
        x1 = F.relu(self.enc_conv1(x))
        x2 = F.relu(self.enc_conv2(x1))
        x3 = self.pool(x2)
        x4 = F.relu(self.enc_conv3(x3))
        x5 = F.relu(self.enc_conv4(x4))
        
        # Decoding
        x6 = self.upsample(x5)
        x7 = F.relu(self.dec_conv1(x6))
        x8 = F.relu(self.dec_conv2(x7))
        x9 = self.dec_conv3(x8)
        
        return x9

class AzFilter_test_v0(nn.Module):
    def __init__(self, length=4096, wavelength=0.055465764662349676):
        super().__init__()

        # Initialize learnable parameters for slant_range_vec and D
        # self.slant_range_vec = nn.Parameter(torch.randn(length))
        # self.D = nn.Parameter(torch.randn(length))
        self.Y = nn.Parameter(torch.randn(length), dtype=torch.complex64)
        # Wavelength is known, so we don't need to make it a learnable parameter
        self.wavelength = wavelength

    def forward(self, x):
        # Calculate the azimuth filter using the learnable parameters and the given equation
        # az_filter = torch.exp(torch.tensor([0+4j], dtype=torch.complex64) * np.pi * self.slant_range_vec * self.D / self.wavelength)
        az_filter = torch.exp(torch.tensor([0+4j], dtype=torch.complex64) * np.pi * self.Y / self.wavelength) 
        # Match the dimensions of az_filter with x if needed, before multiplication
        az_filter = az_filter.view(1, 1, 1, -1)  # Assuming x is [batch, channels, height, width]
        # Apply the azimuth filter through element-wise multiplication
        output = x * az_filter
        return output
##############################################################################

class ModulatedAzimuthChirp(nn.Module):
    def __init__(self):
        super(ModulatedAzimuthChirp, self).__init__()
        
        # Initialize learnable parameters
        self.sample_rate = nn.Parameter(torch.tensor(10.0))  # Sample rate in Hz
        self.duration = nn.Parameter(torch.tensor(800.0))  # Duration in seconds
        self.start_frequency = nn.Parameter(torch.tensor(0.0))  # Starting frequency in Hz
        self.chirp_rate = nn.Parameter(torch.tensor(1e-5))  # Chirp rate in Hz/s
        self.modulation_factor = nn.Parameter(torch.tensor(1.0))  # Frequency modulation factor
        self.shift = nn.Parameter(torch.tensor(0.0))  # Phase shift in seconds

    def forward(self):
        # Generate time array
        t = torch.linspace(-self.duration.item() / 2, self.duration.item() / 2, int(self.sample_rate.item() * self.duration.item()))
        
        # Generate azimuth chirp signal with constant amplitude
        phase = 2 * torch.pi * self.start_frequency * (t - self.shift) + torch.pi * self.chirp_rate * (t - self.shift)**2
        # Add frequency modulation
        phase += self.modulation_factor * torch.sin(torch.pi * (t - self.shift) / self.duration)**2 * torch.sin(2 * torch.pi * self.start_frequency * (t - self.shift))
        
        chirp_signal = torch.exp(1j * phase)
        
        # Apply "oval" amplitude modulation
        amplitude_modulation = torch.cos(torch.pi * (t - self.shift) / self.duration)**2
        chirp_signal *= amplitude_modulation
        
        return t, chirp_signal

class FFTChirpMatrix(nn.Module):
    def __init__(self, num_columns, num_rows):
        super(FFTChirpMatrix, self).__init__()
        
        self.num_columns = num_columns
        self.num_rows = num_rows
        self.chirp_modules = nn.ModuleList([ModulatedAzimuthChirp() for _ in range(num_columns)])
        
    def forward(self):
        fft_matrix = []
        
        for i in range(self.num_columns):
            _, chirp_signal = self.chirp_modules[i]()
            chirp_signal = chirp_signal[:self.num_rows]
            fft_chirp = torch.fft.fft(chirp_signal)
            fft_matrix.append(fft_chirp)
        
        fft_matrix = torch.stack(fft_matrix, dim=1)
        
        return fft_matrix

class robertNet(nn.Module):
    def __init__(self, input_shape: tuple = (1, 2, 4096, 4096)) -> None:
        super().__init__()
        original_range_filter = torch.load('rg_filter.pt').to('cpu').numpy()
        range_filter = self.interpolate_range_filter(original_filter=original_range_filter, new_size=(input_shape[-2], input_shape[-1]), method='zero')
        self.range_filter_fft = self.fft_conjugate(range_filter).to(device)
        
        az_module = FFTChirpMatrix(num_columns=input_shape[-1], num_rows=input_shape[-2])
        self.az_filter = nn.Parameter(az_module())
    
    
    @staticmethod
    def fft_conjugate(range_filter):
        fft_result = torch.fft.fft(torch.tensor(range_filter, dtype=torch.complex64))
        conjugate_result = torch.conj(fft_result)
        return conjugate_result
    
    @staticmethod
    def interpolate_range_filter(original_filter, new_size, method='linear'):
        x = np.linspace(0, len(original_filter) - 1, len(original_filter))
        f = interp1d(x, original_filter, kind=method)
        x_new = np.linspace(0, len(original_filter) - 1, new_size[1])
        return f(x_new)
    
    @staticmethod
    def fft2D(x):
        """
        Perform FFT on radar data along range and azimuth lines.

        Parameters:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width].

        Returns:
            torch.Tensor: Transformed tensor after FFT operations.
        """
        # FFT each range line (along width=range, which is dim=-1)
        x_fft_w = torch.fft.fft(x, dim=-1)
        # FFT each azimuth line (along height=azimuth, which is dim=-2) and shift the zero frequency component to the center
        x_fft_hw = torch.fft.fft(x_fft_w, dim=-2)
        x_fftshift_hw = torch.fft.fftshift(x_fft_hw, dim=[-2, -1])
        return x_fftshift_hw
    
    def _plot_rg_filter(self):
        plt.figure(figsize=(13,6))
        plt.subplot(2, 1, 1)
        plt.plot(self.range_filter_fft.cpu().numpy().real)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        
        plt.subplot(2, 1, 2)
        plt.plot(torch.fft.ifft(self.range_filter_fft).cpu().numpy().real)
        plt.xlabel('Time (s)')
        plt.ylabel('Power')
        plt.show()
    
    @staticmethod
    def _plot_tensor(x):
        if type(x) == torch.Tensor:
            input_tensor = x.cpu().numpy()
        else:
            input_tensor = x
        
        input_data_mean = np.mean(np.abs(input_tensor))
        input_data_std = np.std(np.abs(input_tensor))
        
        plt.figure(figsize=(13,6))
        norm_input = colors.LogNorm(vmin=input_data_mean - input_data_std * 0.5 + 1e-10, vmax=input_data_mean + input_data_std * 2)
        plt.imshow(np.abs(input_tensor), cmap='viridis', norm=norm_input)
        plt.axis('off')
    
    def range_compression(self, x):
        """ Perform range compression on the input tensor. (Range filter is pre-computed) """
        x = x * self.range_filter_fft
        return x
    
    @staticmethod
    def ifft_range(x):
        """ Perform inverse FFT on the input tensor along range lines (dim=-1). """
        x = torch.fft.ifft(x, dim=-1)
        x = torch.fft.fftshift(x, dim=-1)
        return x
    
    def azimuth_compression(self, x):
        az_filter = self.az_filter
        return x * az_filter
    

    def forward(self, x):
        # 2D FFT:
        x = self.fft2D(x)
        # RANGE COMPRESSION:
        x = self.range_compression(x) # data is range compressed
        # TODO: add rcmc filter
        # ...
        
        x = self.ifft_range(x)
        # Azimuth compression:
        x = self.azimuth_compression(x)
        # Back in time:
        x = torch.fft.ifft(x, axis=-2) # time domain, iftt azimuth lines
        return x
    
# 2. Define the Shannon Entropy Loss
def shannon_entropy_loss(I):
    I_abs = torch.abs(I)
    S = torch.sum(I_abs**2)
    D = (I_abs**2) / S
    D_nonzero = torch.where(D > 0, D, torch.tensor(1.0).to(D.device))
    loss = -torch.sum(D * torch.log(D_nonzero))
    return loss

def complex_mse_loss(pred, target):
    assert torch.is_complex(pred), "Predicted tensor is not complex"
    assert torch.is_complex(target), f"Target tensor {target.shape} is not complex"
    
    real_diff = pred.real - target.real
    imag_diff = pred.imag - target.imag
    mse_real = F.mse_loss(real_diff, torch.zeros_like(real_diff))
    mse_imag = F.mse_loss(imag_diff, torch.zeros_like(imag_diff))
    return mse_real + mse_imag

# 3. Use PyTorch Lightning for Training
class FocuserModule(pl.LightningModule):
    def __init__(self, input_img, gt = None):
        super().__init__()

        self.input_img = input_img
        self.gt = gt
        self.model = robertNet(input_shape=input_img.shape)
        self.lr = 1e-2 # learning rate
        if self.gt is not None:
            self.loss_fn = complex_mse_loss
        else:
            self.loss_fn = shannon_entropy_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler,
                    # 'monitor': 'val_loss',  # optional key used for early stopping
                }
        
    def train_dataloader(self):
        assert self.input_img is not None, "Input image tensor is not initialized"
        if self.gt is not None:
            assert self.input_img.shape[0] == self.gt.shape[0], "Size mismatch between input and ground truth"
            dataset = TensorDataset(self.input_img, self.gt)
        else:
            dataset = TensorDataset(self.input_img)
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=7)

    
    def training_step(self, batch, batch_idx):
        img, gt = batch
        output = self.model(img)
        
        if gt is not None:
            loss = self.loss_fn(output, gt)
        else:
            loss = self.loss_fn(output)
    
        self.log('train_loss', loss)
        self.log('lr', self.lr)
        return loss


def read_zarr_database():
    file_path = "Mini_R2F.zarr"
    # To read the root array or group
    root = read_zarr_file(file_path)
    # To read a specific array or group
    raw = read_zarr_file(file_path, "raw")
    gt = read_zarr_file(file_path, "gt")
    return raw, gt


def resize_all():
    radar_data = raw[0]
    radar_data = np.load('radar_data_ifft_numpy.npy')
    radar_data = radar_data[:5000,:5000]
    
    # Convert the image to PyTorch tensor
    input_img_tensor = torch.tensor(radar_data, dtype=torch.complex64)
    real_part = input_img_tensor.real.unsqueeze(0)
    imag_part = input_img_tensor.imag.unsqueeze(0)

    print('shape of resized_real_part: ', real_part.shape)
    print('shape of resized_imag_part: ', imag_part.shape)
    # Resize the image
    resized_real_part = F.interpolate(real_part.unsqueeze(0), scale_factor=0.2, mode='bilinear', align_corners=False).squeeze(0)
    resized_imag_part = F.interpolate(imag_part.unsqueeze(0), scale_factor=0.2, mode='bilinear', align_corners=False).squeeze(0)

    print('shape of resized_real_part: ', resized_real_part.shape)
    print('shape of resized_imag_part: ', resized_imag_part.shape)

    # Combine the real and imaginary parts back into a complex tensor
    resized_img_tensor = torch.complex(resized_real_part.squeeze(0), resized_imag_part.squeeze(0))
    # sar_model = SARModule(resized_img_tensor)
    


if __name__ == '__main__':
    # Usage example
    # idx = 0
    # raw, gt = read_zarr_database()
    # convert to numpy
    # radar_data = torch.tensor(raw[idx], dtype=torch.complex64).to(device)
    # ground_truth = torch.tensor(gt[idx], dtype=torch.complex64).to(device)

    radar_data = torch.rand(1, 2, 512, 512, dtype=torch.complex64).to('cpu')
    ground_truth = torch.rand(1, 2, 512, 512, dtype=torch.complex64).to('cpu')

    # Train using PyTorch Lightning
    sar_model = FocuserModule(radar_data, ground_truth)
    trainer = pl.Trainer(max_epochs=600, log_every_n_steps=1)
    trainer.fit(sar_model)