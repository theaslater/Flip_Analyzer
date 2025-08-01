# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:00:55 2023

@author: Theodora Slater
"""
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import optimize
import sys
import argparse
import os
import pandas as pd
import imageio

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
run = True  # Runs expensive fitting block if set to True, else loads from saved arrays
figures = True  # Determines if the script will create all the figures
second_fitting = False  # Will run a second fitting after cropping data (BROKEN WITHOUT)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process NIfTI file for flip analysis.")
parser.add_argument("-d", "--directory", required=True, help="Directory containing the NIfTI file.")
parser.add_argument("-i", "--image", required=True, help="Name of the NIfTI file.")
parser.add_argument("-b", "--background", required=False, help=".txt file of background cropping data or .nii.gz file of background region.\nBackground.txt is default.")
parser.add_argument("-r", "--run", action='store_false', help="DON'T run the fitting process.")
parser.add_argument("-s", "--site", required=False, help="Site of the experiment, used for fig titles.")
args = parser.parse_args()

# Check if the provided directory exists
if not os.path.isdir(args.directory):
    print(f"Error: The directory '{args.directory}' does not exist.")
    sys.exit(1)
directory = args.directory

# Construct the full file path
file = os.path.join(directory, args.image)

# Check if the NIfTI file exists
if not os.path.isfile(file):
    print(f"Error: The file '{file}' does not exist.")
    sys.exit(1)

#Check if background file exists
background = args.background if args.background else "Background.txt"
background_file = os.path.join(directory, background)
if not os.path.isfile(background_file):
    print(f"Error: The background file '{background_file}' does not exist.")
    sys.exit(1)

# Load the Image NIfTI file

try:
    nii = nib.load(file)
    data = nii.get_fdata()
    header = nii.header
    print("Loaded volume data with dimensions: {}".format(data.shape))
except Exception as e:
    print(f"Error: Unable to load the NIfTI file '{file}'. Ensure it is a valid NIfTI format.")
    sys.exit(1)

# Crop background region from the data
try:
    if background_file.endswith('.txt'):
        with open(background_file, 'r') as f:
            background_crop = [int(x) for x in f.read().strip().split(' ')]
        if len(background_crop) != 8:
            raise ValueError("Background crop file must contain exactly 8 integers.\n x_start, x_end, y_start, y_end, z_start, z_end, t_start, t_end expected.")
        x_start, x_end, y_start, y_end, z_start, z_end, t_start, t_end = background_crop
        background_slices = data[x_start:x_end, y_start:y_end, z_start:z_end, t_start:t_end]
        background_slices = background_slices[:, :, 3, :]  # Extract the relevant slice
        background_mean = np.nanmean(background_slices)  # Calculate the mean of the background region
        print("Background mean value calculated: {}".format(background_mean))
    elif background_file.endswith('.nii') or background_file.endswith('.nii.gz'):
        nii_background = nib.load(background_file)
        background_slices = nii_background.get_fdata()
        if background_slices.ndim != 4:
            raise ValueError("Background file must be a 4D image file.")
        background_mean = np.nanmean(background_slices)
        print("Background mean value calculated from .nifti file: {}".format(background_mean))
    else:
        raise ValueError("Unsupported background file format. Use .txt or .nii/.nii.gz files.")
except Exception as e:
    print(f"Error: Unable to read or process the background crop file '{background_file}'. {e}")
    sys.exit(1)


# Extract specific slices from the data
slices = np.array(data[:, :, 3, :])

print("Reduced to: {}".format(slices.shape))

# Define flip angles
angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 160, 180, 200, 220])

# Initialize arrays for storing results
amp = np.zeros((data.shape[0], data.shape[1]))
omega = np.zeros((data.shape[0], data.shape[1]))
offset = np.zeros((data.shape[0], data.shape[1]))
ffmap = np.zeros((data.shape[0], data.shape[1]))
r_squaredmap = np.zeros((data.shape[0], data.shape[1]))
guess_amp = np.zeros((data.shape[0], data.shape[1]))
guess_offset = np.zeros((data.shape[0], data.shape[1]))
Constant = np.zeros((data.shape[0], data.shape[1]))

# Subtract background signal
slices = slices - background_mean

# # Create a mask to remove regions without signal
# mask = np.zeros([np.shape(slices)[0], np.shape(slices)[1]])
# R = 28  # Radius for the mask
# XOffset = 44
# YOffset = 35
# X = int(R)
# for x in range(-X, X+1):
#     Y = int(abs(R*R - x*x)**0.5)
#     for y in range(-Y, Y+1):
#         mask[x+XOffset, y+YOffset] = 1
# mask[XOffset-R:XOffset+R+1, YOffset-R:45] = 0
# mask[25:61, 7:16] = 1
# mask = mask[:, :, np.newaxis]
# slices = slices * mask

# # TEMPORARY - Set flip angle 0 to background
# slices[:, :, 0] = mask[:, :, 0] * background

# Replace zeros with NaN
slices[slices == 0] = float("nan")

# Fitting sine curves to the data
if args.run:
    print("Fitting sin curves...")
    for i in range(np.shape(slices)[0]):  # Iterate over x-dimension
        for j in range(np.shape(slices)[1]):  # Iterate over y-dimension
            if np.all(slices[i, j, :]) != float("nan"):  # Only process valid voxels MAYBE CHECK LATER
                intensities = slices[i, j, :]  # Extract voxel intensities across angles
                guess_amp[i,j] = np.ptp(intensities)  # Guess amplitude as peak-to-peak difference
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
    if np.all(amp) == float("nan"):
        # If all values are NaN, fitting failed for all voxels
        print("Error: Fitting amplitude failed for all voxels. Please check the input data.")
        sys.exit(1)
    elif np.all(omega) == float("nan"):
        # If all values are NaN, fitting omega failed for all voxels
        print("Error: Fitting omega failed for all voxels. Please check the input data.")
        sys.exit(1)
    elif np.all(r_squaredmap) == float("nan"):
        # If all values are NaN, fitting R-squared failed for all voxels
        print("Error: Fitting R-squared failed for all voxels. Please check the input data.")
        sys.exit(1)
    elif np.all(ffmap) == float("nan"):
        # If all values are NaN, fitting frequency guesses failed for all voxels
        print("Error: Fitting frequency guesses failed for all voxels. Please check the input data.")
        sys.exit(1)
    elif np.all(r_squaredmap) == float("nan"):
        # If all values are NaN, fitting constant failed for all voxels
        print("Error: Fitting r_squareds failed for all voxels. Please check the input data.")
        sys.exit(1)
    np.save(os.path.join(directory, 'amp'), amp)
    np.save(os.path.join(directory,  'omega'), omega)
    np.save(os.path.join(directory, 'frequency_guesses'), ffmap)
    np.save(os.path.join(directory, 'R_Squared_Map'), r_squaredmap)
else:
    try:
        # Load previously saved results
        amp = np.load(os.path.join(directory, 'amp.npy'))
        omega = np.load(os.path.join(directory, 'omega.npy'))
        ffmap = np.load(os.path.join(directory, 'frequency_guesses.npy'))
        r_squaredmap = np.load(os.path.join(directory, 'R_Squared_Map.npy'))
    except FileNotFoundError:
        print("Error: Required files not found. Please run the fitting process first.")
        sys.exit(1)

# Calculate B1 map and normalized omega values
Omega_Normalised = 180 / (np.pi / omega)
B1Map = omega * np.sin(omega * 38)
np.save(os.path.join(directory, 'B1Map_V2'), B1Map)

# Generate figures if enabled
if figures:
    # Map of Omega values
    fig, ax = plt.subplots()
    pos = ax.imshow(omega.T, vmin=0, vmax=0.03, origin='lower')
    fig.colorbar(pos, ax=ax, shrink = 0.75)
    if args.site:
        ax.set_title(f'Map of Omega Values Without Offset - {args.site}')
    else:
        ax.set_title('Map of Omega Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'Map of Omega Values Without Offset'), dpi=300)

    # Map of R-squared values
    fig, ax = plt.subplots()
    pos = ax.imshow(r_squaredmap.T, origin='lower')
    fig.colorbar(pos, ax=ax, shrink = 0.75)
    if args.site:
        ax.set_title(f'Map of R Squared Values Without Offset - {args.site}')
    else:
        ax.set_title('Map of R Squared Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'Map of R Squared Values Without Offset'), dpi=300)

    # Map of Amplitude values
    fig, ax = plt.subplots()
    pos = ax.imshow(amp.T, vmax=5.e6, vmin=1.e5, origin='lower')
    fig.colorbar(pos, ax=ax, shrink = 0.75)
    if args.site:
        ax.set_title(f'Map of Amplitude Values Without Offset - {args.site}')
    else:
        ax.set_title('Map of Amplitude Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'Map of Amplitude Values Without Offset'), dpi=300)

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
    plt.savefig(os.path.join(directory, 'Fits of Curves Without An Offset'), dpi=300)

    # Images at flip angles
    fig, axs = plt.subplots(3, 7, sharex=True, sharey=True)
    if args.site:
        fig.suptitle(f"Images at flip angles - {args.site}", fontsize=18)
    else:
        fig.suptitle("Images at flip angles")
    for i in range(np.shape(slices)[2]):
        axs[i//7, i%7].imshow(slices[:, :, i].T, origin='lower')
        title = 'Flip Angle: ' + str(angles[i])
        axs[i//7, i%7].set_title(title)
    fig.set_size_inches(12, 6)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig(os.path.join(directory, f'Images_at_flip_angles'), dpi=300)
    
    # Create a GIF looping through the images at 0.2 frames per second

    # Collect images at flip angles into a list
    images = []
    for i in range(np.shape(slices)[2]):
        fig, ax = plt.subplots()
        ax.imshow(slices[:, :, i].T, origin='lower')
        title = f'Flip Angle: {angles[i]}'
        ax.set_title(title)
        fig.set_size_inches(6, 6)
        plt.tight_layout()
        
        # Save the current frame to a temporary file
        temp_file = os.path.join(directory, f'temp_frame_{i}.png')
        plt.savefig(temp_file, dpi=100)
        plt.close(fig)
        
        # Append the temporary file to the images list
        images.append(imageio.imread(temp_file))

    # Create the GIF
    gif_path = os.path.join(directory, 'Flip_Angles_Animation.gif')
    imageio.mimsave(gif_path, images, duration=0.2)

    # Clean up temporary files
    for temp_file in [os.path.join(directory, f'temp_frame_{i}.png') for i in range(np.shape(slices)[2])]:
        os.remove(temp_file)

    print(f"GIF saved at {gif_path}")


    # B1 Map
    fig, ax = plt.subplots()
    pos = ax.imshow(Omega_Normalised.T, vmin=0, vmax=3, origin='lower')
    fig.colorbar(pos, ax=ax, shrink = 0.75)
    if args.site:
        ax.set_title(f'Map of B1 Map With a 38 Degree Flip Angle - {args.site}')
    else:
        ax.set_title('Map of B1 Map With a 38 Degree Flip Angle')
    fig.set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'Map of B1 Map With a 38 Degree Flip Angle'), dpi=300)

    fig, ax = plt.subplots()
    pos = ax.imshow(B1Map.T, vmin=0, origin='lower')
    fig.colorbar(pos, ax=ax, shrink = 0.75)
    if args.site:
        ax.set_title(f'Map of Magnetic Field Values Without Offset - {args.site}')
    else:
        ax.set_title('Map of Magnetic Field Values Without Offset')
    fig.set_size_inches(9, 9)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, 'Map of B1 Values Without Offset'), dpi=300)

# Save B1 map as a NIfTI file
B1Map = nib.Nifti1Image(Omega_Normalised, nii.affine)
nib.save(B1Map, os.path.join(directory, 'B1Map_38Deg_Nom.nii.gz'))