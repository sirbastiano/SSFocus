import numpy as np

def compute_shannon_entropy(I):
    """
    Compute the Shannon entropy of a SAR image.
    
    Parameters:
    - I: 2D numpy array (can be complex), where I[q,k] represents the amplitude (or complex value) of the scattering unit at azimuth pulse index q and range cell index k.
    
    Returns:
    - entropy: Shannon entropy of the SAR image.
    """
    # Compute the absolute values if I is complex
    I_abs = np.abs(I)
    
    # Compute the total energy of the image
    S = np.sum(I_abs**2)
    
    # Compute the scattering intensity density of the image
    D = (I_abs**2) / S
    
    # Ensure that we don't have zero values in D for the logarithm
    D_nonzero = np.where(D > 0, D, 1)
    
    # Compute the Shannon entropy
    entropy = -np.sum(D * np.log(D_nonzero))
    
    return entropy