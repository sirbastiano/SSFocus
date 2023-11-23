import matplotlib.pyplot as plt  
from matplotlib.colors import LogNorm
import numpy as np


def plot_with_logscale(res, k=1):
    plt.figure(dpi=300, figsize=(12,12))
    res = np.abs(res.detach().cpu().numpy())
    plt.imshow(res, norm=LogNorm(vmin=res.mean()-k*res.std(), vmax=res.mean()+k*res.std()))  # vmin should be > 0 for LogNorm
    plt.colorbar
    
    
    
def main():
    print("Starting main..")
    pass


if __name__ == '__main__':
    main()