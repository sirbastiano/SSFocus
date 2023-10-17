import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.optim as optim
import unittest


       



class FocusBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Reproduce the behaviour of a Range Focusing Step for SAR data:
        
        def forward(self, x):
            pass
        


class SpectrumAttentionBlock(nn.Module):
    """
    Normalizes the input spectrum by subtracting the learnable means and dividing by the learnable standard deviations.
    Duplicates channel values to match the number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Parameters for the splitter
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # assert that out_channels is a multiple of in_channels:
        assert out_channels % in_channels == 0, "out_channels must be a multiple of in_channels"

        # Initialize the learnable parameters, one for each output channel
        self.learnable_means = nn.Parameter(torch.ones(self.out_channels), requires_grad=True)
        self.learnable_stds = nn.Parameter(torch.ones(self.out_channels), requires_grad=True)
        
    def normalize(self, x):
        # Normalize the input spectrum
        x_normalized = (x - self.learnable_means) / self.learnable_stds
        return x_normalized
        
    def forward(self, x):
        # duplicate channel values to match the number of output channels
        x = x.repeat(1, self.out_channels // self.in_channels)
        x = self.normalize(x)
        return x
            

class ConvBA(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, batch_norm=True, activation=nn.ReLU(inplace=True)):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
        if batch_norm:
            self.add_module('bn', nn.BatchNorm2d(out_channels))
        if activation:
            self.add_module('activation', activation)

# Define the PhiNet model with multi-branch architecture
class PhiNet(nn.Module):
    def __init__(self, conv_channels_list=[2, 32, 64, 128, 256], expand_channels: int = 12, use_SA: bool = False, activation=nn.ReLU(inplace=True)):
        super(PhiNet, self).__init__()
        
        assert len(conv_channels_list) == 5, "conv_channels_list must have 4 elements.."
        self.in_channels = conv_channels_list[0]
        self.out_channels = conv_channels_list[-1]
        self.activation = activation

        # Spectrum Attention Block:
        if use_SA:
            self.expand_channels = expand_channels
            self.attention_spectral = SpectrumAttentionBlock(in_channels=self.in_channels, out_channels=expand_channels) 
        else:
            self.attention_spectral = nn.Identity()
            self.expand_channels = self.in_channels

        # Pooling Branch
        self.pool = nn.MaxPool2d(16, 16)
        self.bn = nn.BatchNorm2d(self.expand_channels)
        
        # Strided Convolution Branch 7x7
        self.conv_branch = self.create_conv_layers(conv_channels_list, batch_norm=True)
        
        # Self Decoder Branch 1x1 (2D Convolution) to output 1 channel:
        self.decoder_pool = nn.Conv2d(self.expand_channels, 1, kernel_size=1, stride=1, padding=0)
        self.decoder_conv = nn.Conv2d(self.out_channels, 1, kernel_size=1, stride=1, padding=0)

    def create_conv_layers(self, channels, batch_norm=True):
        layers = [ConvBA(in_channels=in_c, out_channels=out_c, kernel_size=7, stride=2, padding=3, dilation=1) for in_c, out_c in zip(channels[:-1], channels[1:])]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.attention_spectral(x) # spectrum attention common
        # A) Pooling Branch Forward Pass
        x_pool = self.bn(self.pool(x)) # b, 2 or 12, w, h
        x_pool = self.decoder_pool(x_pool) # b, 1, w, h (same as)
        
        # B) Convolution Branch Forward Pass
        x_conv = self.conv_branch(x) # b, 12, w, h
        x_conv = self.decoder_conv(x_conv) # b, 1, w, h (same as)

        assert x_pool.shape == x_conv.shape, f"Shapes of both branches must be the same.. {x_pool.shape} != {x_conv.shape}"

        # Concatenating the outputs of both branches with addition
        x_out = x_pool + x_conv
        return x_out # b, 1, w//16, h//16

                
        

    
    
    
    
class TestPhiNet(unittest.TestCase):
    
    def setUp(self):
        self.conv_channels_list = [2, 64, 128, 256]
        self.out_channels = 12
        self.model = PhiNet(conv_channels_list=self.conv_channels_list, out_channels=self.out_channels)
        self.input_tensor = torch.randn(5, 1, 128, 128)  # Batch size of 5, 1 input channel, 128x128 image
    
    def test_init(self):
        self.assertEqual(self.model.in_channels, self.conv_channels_list[0])
        self.assertEqual(self.model.out_channels, self.out_channels)
        
        # Check types of sub-modules
        self.assertIsInstance(self.model.attention_spectral, SpectrumAttentionBlock)
        self.assertIsInstance(self.model.pool, nn.MaxPool2d)
        self.assertIsInstance(self.model.bn, nn.BatchNorm2d)
        self.assertIsInstance(self.model.decoder_conv, nn.Conv1d)
        
    def test_forward_pass_shape(self):
        output = self.model(self.input_tensor)
        self.assertEqual(output.shape, torch.Size([5, 1, 8, 8]))  # Change the shape based on your model's expected output shape
        
    def test_branch_addition(self):
        with torch.no_grad():
            attention_output = self.model.attention_spectral(self.input_tensor)
            pool_output = self.model.bn(self.model.pool(attention_output))
            conv_output = self.model.conv_branch(attention_output)
            
            # Apply 1D decoder convolution
            pool_output = self.model.decoder_conv(pool_output)
            conv_output = self.model.decoder_conv(conv_output)
            
            expected_output = pool_output + conv_output
            actual_output = self.model(self.input_tensor)
            
            self.assertTrue(torch.allclose(expected_output, actual_output, atol=1e-5))


class TestNormSpectrumBlock(unittest.TestCase):

    def test_initialization(self):
        # This should work
        SpectrumAttentionBlock(2, 4)
        
        # This should raise an AssertionError
        with self.assertRaises(AssertionError):
            SpectrumAttentionBlock(2, 5)
            
    def test_forward_pass(self):
        model = SpectrumAttentionBlock(2, 4)
        x = torch.rand(5, 2)  # 5 samples, 2 channels
        out = model(x)
        
        # Checking the shape of the output
        self.assertEqual(out.shape, (5, 4))
        
    def test_normalization(self):
        model = SpectrumAttentionBlock(1, 2)
        model.learnable_means.data.fill_(1.0)
        model.learnable_stds.data.fill_(2.0)
        
        # Single channel, single sample, value = 3
        x = torch.tensor([[3.0]])
        out = model(x)
        
        # Output should be (3 - 1) / 2 = 1 for each duplicated channel
        self.assertTrue(torch.allclose(out, torch.tensor([[1.0, 1.0]]), atol=1e-5))
        
    def test_gradient(self):
        model = SpectrumAttentionBlock(2, 4)
        x = torch.rand(5, 2, requires_grad=True)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Forward pass
        out = model(x)
        loss = out.sum()
        
        # Backward pass
        optimizer.zero_grad()



if __name__ == '__main__':
    # unittest.main()

    # Initialize the PhiNet model and print its architecture
    phi_net_model = PhiNet()
    print(phi_net_model)
