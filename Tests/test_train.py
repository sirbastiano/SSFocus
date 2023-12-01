import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

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

# 1. Define the Simple CNN Model
# class ComplexCNN(nn.Module):
#     def __init__(self):
#         super(ComplexCNN, self).__init__()
#         # Convolution only along y direction
#         # Since we're treating real and imaginary parts separately, we have 2 input channels and 2 output channels.
#         self.conv = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(28, 1), padding=(1, 0), bias=False)

#     def forward(self, x):
#         # Assuming x is a complex tensor, separate real and imaginary parts
#         x_real = x.real
#         x_imag = x.imag

#         # Stack them into a new tensor such that the real and imaginary parts are separate channels
#         x_stacked = torch.stack((x_real, x_imag), dim=1)

#         # Apply convolution
#         x_stacked = self.conv(x_stacked)
#         x_stacked = F.relu(x_stacked)
#         x_stacked = self.conv(x_stacked)
#         x_stacked = F.relu(x_stacked)
                
#         # Reconstruct the complex tensor from the real and imaginary parts in the output channels
#         out_real = x_stacked[:, 0]
#         out_imag = x_stacked[:, 1]

#         # Return as a complex tensor
#         return torch.complex(out_real, out_imag)

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


# dtype=torch.complex64

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


class AzFilter(nn.Module):
    def __init__(self, length=4096, wavelength=0.055465764662349676):
        super(AzFilter, self).__init__()

        # Initialize learnable parameters for slant_range_vec and D
        self.slant_range_vec = nn.Parameter(torch.randn(length))
        self.D = nn.Parameter(torch.randn(length))

        # Wavelength is known, so we don't need to make it a learnable parameter
        self.wavelength = wavelength

    def forward(self, x):
        # Calculate the azimuth filter using the learnable parameters and the given equation
        az_filter = torch.exp(torch.tensor([0+4j], dtype=torch.complex64) * np.pi * self.slant_range_vec * self.D / self.wavelength)
        
        # Match the dimensions of az_filter with x if needed, before multiplication
        az_filter = az_filter.view(1, 1, 1, -1)  # Assuming x is [batch, channels, height, width]

        # Apply the azimuth filter through element-wise multiplication
        output = x * az_filter

        return output

class robertNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.range_filter_fft = torch.load('range_filter.pt')
    
    
    
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
        x_fft_w = torch.fft.fft(x, signal_ndim=1, normalized=False, dim=-1)
        # FFT each azimuth line (along height=azimuth, which is dim=-2) and shift the zero frequency component to the center
        x_fft_hw = torch.fft.fft(x_fft_w, signal_ndim=1, normalized=False, dim=-2)
        x_fftshift_hw = torch.fft.fftshift(x_fft_hw, dim=-2)
        return x_fftshift_hw
    
    @staticmethod
    def range_compression(self, x):
        """ Perform range compression on the input tensor. (Range filter is pre-computed) """
        x = x * self.range_filter_fft
        return x
    
    @staticmethod
    def ifft_range(x):
        """ Perform inverse FFT on the input tensor along range lines (dim=-1). """
        x = torch.fft.ifft(x, signal_ndim=1, normalized=False, dim=-1)
        x = torch.fft.fftshift(x, dim=-1)
        return x
    
    def forward(self, x):
        # Separate real and imaginary parts
        
        x = self.fft2D(x)
        x = self.range_compression(x) # data is range compressed
        x = self.ifft_range(x)
        
        
        x = torch.fft.ifft(x, axis=-2) # time domain
        return x
    
    
    

# 2. Define the Shannon Entropy Loss
def shannon_entropy_loss(I):
    I_abs = torch.abs(I)
    S = torch.sum(I_abs**2)
    D = (I_abs**2) / S
    D_nonzero = torch.where(D > 0, D, torch.tensor(1.0).to(D.device))
    loss = -torch.sum(D * torch.log(D_nonzero))
    return loss

# 3. Use PyTorch Lightning for Training
class SARModule(pl.LightningModule):
    def __init__(self, input_img):
        super(SARModule, self).__init__()
        # self.model = ComplexCNN()
        self.model = UNet(2, 2)
        self.input_img = input_img
        self.lr = 1e-3 # learning rate

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
        return torch.utils.data.DataLoader([self.input_img], batch_size=1, shuffle=True, num_workers=7)
    
    def training_step(self, batch, batch_idx):
        img = batch
        output = self.model(img)
        loss = shannon_entropy_loss(output)
        self.log('train_loss', loss)
        return loss




if __name__ == '__main__':
    # Usage example
    # file_path = "Mini_R2F.zarr"

    # # To read the root array or group
    # root = read_zarr_file(file_path)

    # # To read a specific array or group
    # raw = read_zarr_file(file_path, "raw")
    # gt = read_zarr_file(file_path, "gt")
    
    # radar_data = raw[0]
    radar_data = np.load('radar_data_ifft_numpy.npy')

    radar_data = radar_data[:5000,:5000]
    
    # # Convert the image to PyTorch tensor
    # input_img_tensor = torch.tensor(radar_data, dtype=torch.complex64)
    # real_part = input_img_tensor.real.unsqueeze(0)
    # imag_part = input_img_tensor.imag.unsqueeze(0)

    # print('shape of resized_real_part: ', real_part.shape)
    # print('shape of resized_imag_part: ', imag_part.shape)
    # # Resize the image
    # resized_real_part = F.interpolate(real_part.unsqueeze(0), scale_factor=0.2, mode='bilinear', align_corners=False).squeeze(0)
    # resized_imag_part = F.interpolate(imag_part.unsqueeze(0), scale_factor=0.2, mode='bilinear', align_corners=False).squeeze(0)

    # print('shape of resized_real_part: ', resized_real_part.shape)
    # print('shape of resized_imag_part: ', resized_imag_part.shape)

    # # Combine the real and imaginary parts back into a complex tensor
    # resized_img_tensor = torch.complex(resized_real_part.squeeze(0), resized_imag_part.squeeze(0))

    # Train using PyTorch Lightning
    # sar_model = SARModule(resized_img_tensor)
    sar_model = SARModule(radar_data)
    trainer = pl.Trainer(max_epochs=30, log_every_n_steps=1)
    trainer.fit(sar_model)