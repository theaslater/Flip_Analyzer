# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:00:55 2023

@author: Theodora Slater
"""
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import optimize

plt.close('all');

# Define a sine function for curve fitting
def sinfunc(t, A, w):  
    return abs(A * np.sin(w*t))

# Calculate R-squared value for the fit
def rsquared(x, y, *args):
    residuals = y - sinfunc(x, *args)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return (1 - (ss_res / ss_tot))

# Flags to control script behavior
run = False  # Runs expensive fitting block if set to True, else loads from saved arrays
figures = True  # Determines if the script will create all the figures
second_fitting = False  # Will run a second fitting after cropping data (BROKEN WITHOUT)

# Load the NIfTI file
file = "FlipAngleSweep3_WIP_FLASHSODIUM_RL_Experiment1_20230526150236_201.nii"
nii = nib.load(file)
data = nii.get_fdata()
header = nii.header
print("Loaded volume data with dimensions: {}".format(data.shape))

# Extract specific slices from the data
slices = data[:, :, 3, :]
print("Reduced to: {}".format(slices.shape))

# Define flip angles
angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 160, 180, 200, 220])

# Initialize arrays for storing results
amp, omega, offset, ffmap, r_squaredmap, guess_amp, guess_offset = (np.zeros([np.shape(data)[0], np.shape(data)[1]]) for i in range(7))
C = np.zeros([np.shape(data)[0], np.shape(data)[1]])

# Subtract background signal
background = np.average(slices[0:96, 18:40, :].T)
slices = slices - background

# Create a mask to remove regions without signal
mask = np.zeros([np.shape(slices)[0], np.shape(slices)[1]])
R = 28  # Radius for the mask
XOffset = 44
YOffset = 35
X = int(R)
for x in range(-X, X+1):
    Y = int(abs(R*R - x*x)**0.5)
    for y in range(-Y, Y+1):
        mask[x+XOffset, y+YOffset] = 1
mask[XOffset-R:XOffset+R+1, YOffset-R:45] = 0
mask[25:61, 7:16] = 1
mask = mask[:, :, np.newaxis]
slices = slices * mask

# TEMPORARY - Set flip angle 0 to background
slices[:, :, 0] = mask[:, :, 0] * background

# Replace zeros with NaN
slices[slices == 0] = float("nan")

# Fitting sine curves to the data
if run:
    print("Fitting sin curves...")
    for i in range(np.shape(slices)[0]):  # Iterate over x-dimension
        for j in range(np.shape(slices)[1]):  # Iterate over y-dimension
            if np.all(slices[i, j, :]) != float("nan"):  # Only process valid voxels
                intensities = slices[i, j, :]  # Extract voxel intensities across angles
                guess_amp[i, j] = np.ptp(intensities)  # Guess amplitude as peak-to-peak difference
                r_squareds = np.zeros(np.shape(angles))
                amps = np.zeros(np.shape(angles))
                omegas = np.zeros(np.shape(angles))
                try:
                    first_fits = np.zeros((3, 7))  # Store initial fits
                    for k in range(np.shape(first_fits)[1]):  # Iterate over guesses
                        guess_freq = np.pi / ((k*20) + 80)
                        guess = np.array([guess_amp[i, j], guess_freq])  # Initial guess
                        popt, pcov = optimize.curve_fit(sinfunc, angles, intensities, p0=guess, maxfev=1000)
                        first_fits[0, k] = abs(popt[0])  # Amplitude
                        first_fits[1, k] = abs(popt[1])  # Omega
                        first_fits[2, k] = rsquared(angles, intensities, *popt)  # R-squared
                    best_index = np.argmax(first_fits, axis=1)[2]  # Select best fit based on R-squared
                    if second_fitting:
                        minimum_angle = int(np.pi / first_fits[1, best_index])  # First minimum point
                        new_angles = angles[angles < minimum_angle]  # Crop angles
                        new_intensities = intensities[0:np.shape(new_angles)[0]]  # Crop intensities
                        popt, pcov = optimize.curve_fit(sinfunc, new_angles, new_intensities, 
                                                        p0=[first_fits[0, best_index], first_fits[1, best_index]], 
                                                        maxfev=1000)
                        amp[i, j], omega[i, j] = popt
                        r_squaredmap[i, j] = rsquared(new_angles, new_intensities, *popt)
                    else:
                        amp[i, j], omega[i, j] = [first_fits[0, best_index], first_fits[1, best_index]]
                        r_squaredmap[i, j] = first_fits[2, best_index]
                except:
                    # Handle cases where fitting fails
                    r_squaredmap[i, j] = float("nan")
                    amp[i, j] = float("nan")
                    omega[i, j] = float("nan")
            else:
                # Set invalid voxels to NaN
                r_squaredmap[i, j] = float("nan")
                amp[i, j] = float("nan")
                omega[i, j] = float("nan")
    print("Sin curves fit")
    # Save results for future use
    np.save('amp', amp)
    np.save('omega', omega)
    np.save('frequency guesses', ffmap)
    np.save('R Squared Map', r_squaredmap)
else:
    # Load previously saved results
    amp = np.load('amp.npy')
    omega = np.load('omega.npy')
    ffmap = np.load('frequency guesses.npy')
    r_squaredmap = np.load('R Squared Map.npy')

# Calculate B1 map and normalized omega values
Omega_Normalised = 180 / (np.pi / omega)
B1Map = omega * np.sin(omega * 38)
np.save('B1Map_V2', B1Map)

# Generate figures if enabled
if figures:
    # Map of Omega values
    fig, ax = plt.subplots()
    pos = ax.imshow(omega.T, vmin=0, vmax=0.03)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of Omega Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.savefig('Map of Omega Values Without Offset', dpi=300)
    plt.show()

    # Map of R-squared values
    fig, ax = plt.subplots()
    pos = ax.imshow(r_squaredmap.T)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of R Squared Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.savefig('Map of R Squared Values Without Offset', dpi=300)
    plt.show()

    # Map of Amplitude values
    fig, ax = plt.subplots()
    pos = ax.imshow(amp.T, vmax=5.e6, vmin=1.e5)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of Amplitude Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.savefig('Map of Amplitude Values Without Offset', dpi=300)
    plt.show()

    # Fits for voxels and their curves
    starting_voxel_x = 41
    starting_voxel_y = 45
    curve_x = np.linspace(0, 260, 82)
    fig, axs = plt.subplots(4, 4, sharex=True)
    fig.suptitle("Fits of Curves At Various Points On Map")
    for i in range(16):
        curve_y = abs(amp[starting_voxel_x, starting_voxel_y+i] * np.sin(curve_x * omega[starting_voxel_x, starting_voxel_y+i]))
        axs[i//4, i%4].plot(curve_x, curve_y, color='blue')
        axs[i//4, i%4].scatter(angles, slices[starting_voxel_x, starting_voxel_y+i, :])
        axs[i//4, i%4].annotate('Omega: ' + str(round(omega[starting_voxel_x, starting_voxel_y+i], 4)), 
                               xy=(0.2, 0), xycoords='axes fraction',
                               xytext=(-20, 20), textcoords='offset pixels',
                               horizontalalignment='left',
                               verticalalignment='bottom')
        title = 'Point Coordinates: [' + str(starting_voxel_x) + ',' + str(starting_voxel_y+i) + ']'
        axs[i//4, i%4].set_title(title)
    fig.set_size_inches(16, 9)
    plt.savefig('Fits of Curves Without An Offset', dpi=300)
    plt.show()

    # Images at flip angles
    fig, axs = plt.subplots(4, 5, sharex=True, sharey=True)
    fig.suptitle("Images at flip angles")
    for i in range(np.shape(slices)[2]):
        axs[i//5, i%5].imshow(slices[:, :, i].T)
        title = 'Flip Angle: ' + str(angles[i])
        axs[i//5, i%5].set_title(title)
        plt.show()

    # B1 Map
    fig, ax = plt.subplots()
    pos = ax.imshow(Omega_Normalised.T)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of Normalised Omega Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.savefig('Map of Normalised Omega Values Without Offset', dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    pos = ax.imshow(B1Map.T)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of Magnetic Field Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.savefig('Map of B1 Values Without Offset', dpi=300)
    plt.show()

# Save B1 map as a NIfTI file
B1Map = nib.Nifti1Image(Omega_Normalised, nii.affine)
nib.save(B1Map, 'B1Map_38Deg_Nom.nii')