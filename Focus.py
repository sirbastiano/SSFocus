# pip install git+https://github.com/Rich-Hall/sentinel1decoder

import matplotlib.pyplot as plt
import cmath
import numpy as np
import math as math
import os
import sentinel1decoder
import logging
from scipy.interpolate import interp1d

root_dir = os.path.join(os.getcwd(), 'L0')
filepath = root_dir + "/02_S1B_S6_RAW__0SDH_20210919T002614_20210919T002640_028760_036EA1_4726.SAFE/1/"
filename = "s1b-s6-raw-s-hh-20210919t002614-20210919t002640-028760-036ea1.dat"
inputfile = filepath+filename

decoder = sentinel1decoder.Level0Decoder(inputfile, log_level=logging.WARNING)
df = decoder.decode_metadata()
print(df)

ephemeris = sentinel1decoder.utilities.read_subcommed_data(df)
print(ephemeris)

selection = df[df["Swath Number"] == 6]
print(selection["BAQ Mode"])
selection = selection.iloc[2000:4000]  # from 61 it works


iq_array = decoder.decode_file(selection)
print("Raw data shape: ", iq_array.shape)
# Image sizes
len_range_line = iq_array.shape[1]
len_az_line = iq_array.shape[0]

# Extract necessary parameters, see both reference documents.
# All necessary parameters are included into the dataframe
c = sentinel1decoder.constants.speed_of_light
TXPL = selection["Tx Pulse Length"].unique()[0]
print("Tx Pulse length", TXPL)
TXPSF = selection["Tx Pulse Start Frequency"].unique()[0]
print("Tx Pulse Start Freq", TXPSF)
TXPRR = selection["Tx Ramp Rate"].unique()[0]
print("Tx Ramp Rate", TXPRR)
RGDEC = selection["Range Decimation"].unique()[0]
print("Range Decimation", RGDEC)
PRI = selection["PRI"].unique()[0]
print("PRI", RGDEC)
rank = selection["Rank"].unique()[0]
print("Rank", rank)
suppressed_data_time = 320/(8*sentinel1decoder.constants.f_ref)  # see pag. 82 of the reference document (Airbus)
print(suppressed_data_time)
range_start_time = selection["SWST"].unique()[0] + suppressed_data_time
wavelength = c / 5.405e9

# Sample rates
range_sample_freq = sentinel1decoder.utilities.range_dec_to_sample_rate(RGDEC)
range_sample_period = 1 / range_sample_freq
az_sample_freq = 1 / PRI
az_sample_period = PRI

# Fast time vector [s] - defines the time axis along the fast time direction
range_line_num = [i for i in range(len_range_line)]
# fast_time = []
# for i in range_line_num:
#     fast_time.append(range_start_time + i * range_sample_period)
fast_time = [range_start_time + i * range_sample_period for i in range_line_num]


# Slant range vector - defines R0, the range of the closest approach for each range cell (i.e. the slant range when
# the radar is closest to the target)

# slant_range = []
# for t in fast_time:
#     slant_range.append((rank * PRI + t) * c / 2)
slant_range = [(rank * PRI + t) * c / 2 for t in fast_time]


# Axes - defines the frequency axes in each direction after FFT
SWL = len_range_line / range_sample_freq
az_freq_vals = np.arange(-az_sample_freq / 2, az_sample_freq / 2, 1 / (PRI * len_az_line))
range_freq_vals = np.arange(-range_sample_freq / 2, range_sample_freq / 2, 1 / SWL)


# We need two parameters which vary over range and azimuth, so we're going to loop over these once
# D is the cosine of the instantaneous squint angle and is defined by the letter D in most literature
# Define a function to calculate D, then apply it inside the loop
def d(range_freq, velocity):
    return math.sqrt(1 - ((wavelength ** 2 * range_freq ** 2) / (4 * velocity ** 2)))


D = np.zeros((len_az_line, len_range_line))

# Spacecraft velocity - numerical calculation of the effective spacecraft velocity
ecef_vels = ephemeris.apply(lambda x: math.sqrt(
    x["X-axis velocity ECEF"] ** 2 + x["Y-axis velocity ECEF"] ** 2 + x["Z-axis velocity ECEF"] ** 2), axis=1)
velocity_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ecef_vels.unique(),
                           fill_value="extrapolate")
x_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ephemeris["X-axis position ECEF"].unique(),
                    fill_value="extrapolate")
y_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ephemeris["Y-axis position ECEF"].unique(),
                    fill_value="extrapolate")
z_interp = interp1d(ephemeris["POD Solution Data Timestamp"].unique(), ephemeris["Z-axis position ECEF"].unique(),
                    fill_value="extrapolate")
space_velocities = selection.apply(lambda x: velocity_interp(x["Coarse Time"] + x["Fine Time"]), axis=1)

x_positions = selection.apply(lambda x: x_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_list()
y_positions = selection.apply(lambda x: y_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_list()
z_positions = selection.apply(lambda x: z_interp(x["Coarse Time"] + x["Fine Time"]), axis=1).to_list()

a = 6378137 # WGS84 semi major axis
b = 6356752.3142 # WGS84 semi minor axis
velocities = np.zeros((len_az_line, len_range_line))

# Now loop over range and azimuth, and calculate spacecraft velocity and D
for i in range(len_az_line):
    H = math.sqrt(x_positions[i]**2 + y_positions[i]**2 + z_positions[i]**2)
    W = float(space_velocities.iloc[i])/H
    lat = math.atan(z_positions[i] / x_positions[i])
    local_earth_rad = math.sqrt(((a**2 * math.cos(lat))**2 + (b**2 * math.sin(lat))**2) / ((a * math.cos(lat))**2 + (b * math.sin(lat))**2))
    for j in range(len_range_line):
        cos_beta = (local_earth_rad**2 + H**2 - slant_range[j]**2) / (2 * local_earth_rad * H)
        this_ground_velocity = local_earth_rad * W * cos_beta
        velocities[i, j] = math.sqrt(float(space_velocities.iloc[i]) * this_ground_velocity)
        D[i, j] = d(az_freq_vals[i], velocities[i, j])

freq_domain_data = np.zeros((len_az_line, len_range_line), dtype=complex)

for az_index in range(len_az_line):
    range_line = iq_array[az_index, :]
    range_fft = np.fft.fft(range_line)
    freq_domain_data[az_index, :] = range_fft

for range_index in range(len_range_line):
    az_line = freq_domain_data[:, range_index]
    az_fft = np.fft.fft(az_line)
    az_fft = np.fft.fftshift(az_fft)
    freq_domain_data[:, range_index] = az_fft

# Create range filter
num_tx_vals = int(TXPL*range_sample_freq)
tx_replica_time_vals = np.linspace(-TXPL/2, TXPL/2, num=num_tx_vals)
phi1 = TXPSF + TXPRR*TXPL/2
phi2 = TXPRR/2
tx_replica = np.zeros(num_tx_vals, dtype=complex)
for i in range(num_tx_vals):
    tx_replica[i] = cmath.exp(2j * cmath.pi * (phi1*tx_replica_time_vals[i] + phi2*tx_replica_time_vals[i]**2))

range_filter = np.zeros(len_range_line, dtype=complex)
index_start = np.ceil((len_range_line-num_tx_vals)/2)-1
index_end = num_tx_vals+np.ceil((len_range_line-num_tx_vals)/2)-2
range_filter[int(index_start):int(index_end+1)] = tx_replica

range_filter = np.fft.fft(range_filter)
range_filter = np.conjugate(range_filter)

for az_index in range(len_az_line):
    freq_domain_data[az_index, :] = freq_domain_data[az_index, :]*range_filter


rcmc_filt = np.zeros(len_range_line, dtype=complex)
range_freq_vals = np.linspace(-range_sample_freq/2, range_sample_freq/2, num=len_range_line)
for az_index in range(len_az_line):
    rcmc_filt = np.zeros(len_range_line, dtype=complex)
    for range_index in range(len_range_line):
        rcmc_shift = slant_range[0]*((1/D[az_index, range_index])-1)
        rcmc_filt[range_index] = cmath.exp(4j * cmath.pi * range_freq_vals[range_index] * rcmc_shift / c)
    freq_domain_data[az_index, :] = freq_domain_data[az_index, :]*rcmc_filt

range_doppler_data = np.zeros((len_az_line, len_range_line), dtype=complex)
for range_line_index in range(len_az_line):
    ifft = np.fft.ifft(freq_domain_data[range_line_index, :])
    ifft_sorted = np.fft.ifftshift(ifft)
    range_doppler_data[range_line_index, :] = ifft_sorted

# Create azimuth filter
az_compressed_data = np.zeros((len_az_line, len_range_line), 'complex')

for az_line_index in range(len_range_line):
    d_vector = np.zeros(len_az_line)
    this_az_filter = np.zeros(len_az_line, 'complex')
    print(this_az_filter.shape)
    print(len(slant_range))
    print(D.shape)
    print(az_freq_vals.shape)
    for i in range(len(az_freq_vals)-1):  # -1
        this_az_filter[i] = cmath.exp((4j * cmath.pi * slant_range[i] * D[i, az_line_index]) / wavelength)
    result = range_doppler_data[:, az_line_index] * this_az_filter[:]
    result = np.fft.ifft(result)
    az_compressed_data[:, az_line_index] = result

# Plot final image
plt.figure(figsize=(16,100))
plt.title("Sentinel-1 Processed SAR Image")
plt.imshow(abs(az_compressed_data[:,:]), vmin=0, vmax=2000, origin='lower')
plt.xlabel("Down Range (samples)")
plt.ylabel("Cross Range (samples)")
plt.show()
