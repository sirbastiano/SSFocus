import torch 
from torch import nn 
import torch.nn.functional as F


def contrast_sharpness_loss(image):
    """
    Calculate a loss that maximizes contrast and sharpness for an image tensor.
    :param image: torch tensor representing the image.
    :return: loss value (lower is better).
    """
    image = rect_im(image)
    # Calculate the magnitude of the complex tensor
    magnitude = torch.sqrt(image.real**2 + image.imag**2)

    # Normalize the magnitude image
    normalized_image = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    # Calculate contrast as the standard deviation of pixel intensities
    contrast = torch.std(normalized_image)

    # Calculate gradients using Sobel filter for sharpness
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    if image.is_cuda:
        sobel_x = sobel_x.cuda()
        sobel_y = sobel_y.cuda()

    grad_x = F.conv2d(normalized_image, sobel_x, padding=1)
    grad_y = F.conv2d(normalized_image, sobel_y, padding=1)

    sharpness = torch.mean(torch.sqrt(grad_x**2 + grad_y**2))

    # Minimize negative of sum of contrast and sharpness
    loss = -(contrast + sharpness)

    return loss


def custom_loss(I):
    alfa = 0.5
    beta = 0.5
    loss = torch.exp(alfa * contrast_sharpness_loss(I) + beta * shannon_entropy_loss(I))
    loss = 0.01 * loss
    return loss
    
    

def rect_im(I):
    # b, c, w, h = I.shape
    # cx, cy = w//2, h//2
    # delta = h//2
    # # compute on the small rectangle
    # I = I[:,:, cy-delta:cy+delta, cx-delta:cx+delta]
    return I

def shannon_entropy_loss(I):
    I = rect_im(I)
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

def AF_loss(pred, target):
    """ Return the autofocus loss """
    alfa = 0.5
    beta = 1.5
    L = alfa * shannon_entropy_loss(pred) + beta * complex_mse_loss(pred, target)




def main():
    print("Starting main program")

if __name__ == '__main__':
    
    main()