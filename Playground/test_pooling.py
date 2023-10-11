import time
import torch  
from torch import nn

# Create a random input tensor simulating a batch of 16 RGB images of size 256x256
input_tensor = torch.randn(16, 3, 256, 256)

# Define the two different pooling approaches
four_pooling_layers = nn.Sequential(
    nn.MaxPool2d(2, 2),
    nn.MaxPool2d(2, 2),
    nn.MaxPool2d(2, 2),
    nn.MaxPool2d(2, 2)
)

one_large_pooling_layer = nn.MaxPool2d(16, 16)

# Measure execution time for 4 pooling layers of size 2x2
start_time_four_pooling = time.time()
output_four_pooling = four_pooling_layers(input_tensor)
elapsed_time_four_pooling = time.time() - start_time_four_pooling

# Measure execution time for 1 pooling layer of size 16x16
start_time_one_large_pooling = time.time()
output_one_large_pooling = one_large_pooling_layer(input_tensor)
elapsed_time_one_large_pooling = time.time() - start_time_one_large_pooling

print(elapsed_time_four_pooling, elapsed_time_one_large_pooling)



############### MACOS CPU
# 0.008565902709960938 0.001341104507446289

############### GOOGLE COLAB GPU
# 0.04514670372009277 0.008213043212890625