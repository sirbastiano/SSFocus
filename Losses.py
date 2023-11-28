import torch 
from torch import nn 
import torch.nn.functional as F


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

def AF_loss(pred, target):
    """ Return the autofocus loss """
    alfa = 0.5
    beta = 1.5
    L = alfa * shannon_entropy_loss(pred) + beta * complex_mse_loss(pred, target)




def main():
    print("Starting main program")

if __name__ == '__main__':
    
    main()