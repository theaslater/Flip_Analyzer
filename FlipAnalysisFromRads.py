# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:00:55 2023

@author: Theodora Slater
"""
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import optimize
import matplotlib.ticker as mticker

plt.close('all');

def sinfunc(t, A, w):  return abs(A * np.sin(w*t))
def rsquared(x,y,*args):
    residuals = y - sinfunc(x, *args)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y-np.mean(y))**2)
    return (1 - (ss_res / ss_tot))

run = True # Runs expensive fitting block if set to true, else will load from saved arrays
figures = True # Determines if the script will create all the figures it can
second_fitting = True # Will run a second fitting after cropping data - BROKEN WITHOUT


file = "FlipAngleSweep3_WIP_FLASHSODIUM_RL_Experiment1_20230526150236_201.nii"
nii=nib.load(file)
data = nii.get_fdata()
header = nii.header
print("Loaded volume data with dimensions: {}".format(data.shape))
# print(header)

slices=data[:,:,3,:]
print("Reduced to: {}".format(slices.shape))
angles=np.array([0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,160,180,200,220])
amp, omega, offset, ffmap, r_squaredmap, guess_amp, guess_offset = (np.zeros([np.shape(data)[0],np.shape(data)[1]]) for i in range(7))
C=np.zeros([np.shape(data)[0],np.shape(data)[1]])
background=np.average(slices[0:96,18:40,:].T)
slices=slices-(background)

# Removing all regions without signal
mask=np.zeros([np.shape(slices)[0],np.shape(slices)[1]])
R=28#22
XOffset=44
YOffset=35
X = int(R) # R is the radius
for x in range(-X,X+1):
    Y = int(abs(R*R-x*x)**0.5)
    for y in range(-Y,Y+1):
        mask[x+XOffset,y+YOffset]=1
mask[XOffset-R:XOffset+R+1,YOffset-R:45]=0
mask[25:61,7:16]=1
mask=mask[:,:,np.newaxis]
slices=slices*mask

# TEMPORARY - CHECKING TO SEE IF SETTING FLIP ANGLE 0 TO BACKGROUND FIXES ISSUES
slices[:,:,0]=mask[:,:,0]*background
# END OF TEMP
slices[slices==0]=float("nan")



# Fitting All The Sine Curves
if run == True:
    print("Fitting sin curves...")
    for i in range(np.shape(slices)[0]): # Sidenote - Would be good to do this without double loop
        for j in range(np.shape(slices)[1]):
            if np.all(slices[i,j,:])!=float("nan"): # Only looking at voxels where there are no 'nan's on all the angles              
                intensities=slices[i,j,:] # Loading row of voxels in across all angles
                guess_amp[i,j] = np.ptp(intensities) # Guess amplitude is the difference between maximum and minimum values
                # Set all variable arrays to 0 to avoid outputs from previous runs being used
                r_squareds=np.zeros(np.shape(angles))
                amps=np.zeros(np.shape(angles))
                omegas=np.zeros(np.shape(angles))
                try: # If the below fitting function doesn't find a fit, the script wont drop out
                    first_fits=np.zeros((3,7))
                    for k in range(np.shape(first_fits)[1]): # This for loop should be outside the try but oh well
                        guess_freq=np.pi/((k*20)+80)
                        guess = np.array([guess_amp[i,j], guess_freq]) # Load all guesses into an array
                        popt, pcov = optimize.curve_fit(sinfunc, angles, intensities, p0=guess, maxfev=1000) # Fitting function
                        first_fits[0,k]=abs(popt[0])#Amp
                        first_fits[1,k]=abs(popt[1])#Omega
                        first_fits[2,k]=rsquared(angles, intensities, *popt) #RSquared
                    best_index=np.argmax(first_fits,axis=1)[2] # Returns index where rsquared is highest
                    if second_fitting==True:
                        minimum_angle=int(np.pi / first_fits[1,best_index]) # Angle of first minimum point
                        new_angles = angles[angles<minimum_angle] # Crops the angles being read at the fist minimum
                        new_intensities=intensities[0:np.shape(new_angles)[0]] # Crops the intensities to the same length
                        popt, pcov = optimize.curve_fit(sinfunc, new_angles, new_intensities, p0=[first_fits[0,best_index],first_fits[1,best_index]], maxfev=1000) # Second Fitting function
                        amp[i,j], omega[i,j]=popt
                        r_squaredmap[i,j]=rsquared(new_angles, new_intensities, *popt)
                    else:
                        amp[i,j], omega[i,j] = [first_fits[0,best_index],first_fits[1,best_index]]
                        r_squaredmap[i,j]=first_fits[2,best_index]
                except: # In event a good fit can't be found, set all variables to 'nan'
                    r_squaredmap[i,j]=float("nan")
                    amp[i,j]=float("nan")
                    omega[i,j]=float("nan")  
            else: # If a data point is missing or voxels are outside of ROI, the voxel on the map is set to 'nan'
                r_squaredmap[i,j]=float("nan")
                amp[i,j]=float("nan") # Sometimes the amplutude is negative, doesn't matter but keeping it positive for neatness
                omega[i,j]=float("nan")   
    print("Sin curves fit")
    #Saving all the variables for faster running in the future
    np.save('amp', amp)
    np.save('omega', omega)
    np.save('frequency guesses', ffmap)
    np.save('R Squared Map',r_squaredmap)
    # np.save('guess_amp',guess_amp)
    # np.save('guess_offset',guess_offset)
else:
   # Load all previous saved arrays if the fitting algorithm hasn't been changed 
   amp = np.load('amp.npy')
   omega = np.load('omega.npy')
   ffmap=np.load('frequency guesses.npy')
   r_squaredmap=np.load('R Squared Map.npy')
   # guess_amp=np.load('guess_amp.npy')
   # guess_offset=np.load('guess_offset.npy')
   
# B1 Map where values are a % of flip angle    
Omega_Normalised = 180/(np.pi/omega)
B1Map = ((omega * np.sin(omega * 38))*(180/(2*np.pi)))*100
np.save('B1Map_V2', B1Map)

#Figures:
if figures == True:
    #Map of Omegas    
    fig, ax = plt.subplots()
    pos=ax.imshow(omega.T, vmin = 0, vmax = 0.03)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of Omega Values Without Offset')
    fig.set_size_inches(9, 9) # set figure's size manually
    plt.savefig('Map of Omega Values Without Offset', dpi=300)
    plt.show()
    
    #Map of R squareds
    fig, ax = plt.subplots()
    pos=ax.imshow(r_squaredmap.T)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of R Squared Values Without Offset')
    fig.set_size_inches(9, 9) # set figure's size manually
    plt.savefig('Map of R Squared Values Without Offset', dpi=300)
    plt.show()
    
    #Map of Amps
    fig, ax = plt.subplots()
    pos=ax.imshow(amp.T, vmax=5.e6, vmin=1.e5)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of Amplitude Values Without Offset')
    fig.set_size_inches(9, 9) # set figure's size manually
    plt.savefig('Map of Amplitude Values Without Offset', dpi=300)
    plt.show()
    
    #Fits for voxels and their curves
    starting_voxel_x = 41
    starting_voxel_y = 45
    curve_x=np.linspace(0,260,82)
    fig, axs=plt.subplots(4, 4, sharex = True)
    fig.suptitle("Fits of Curves At Various Points On Map")
    for i in range(16):
        curve_y=abs(amp[starting_voxel_x,starting_voxel_y+i]*np.sin(curve_x*omega[starting_voxel_x,starting_voxel_y+i]))
        axs[i//4,i%4].plot(curve_x,curve_y,color='blue')
        #axs[i//6,i%6].plot(curve_x,curve_y_Guess,color='orange')
        axs[i//4,i%4].scatter(angles,slices[starting_voxel_x,starting_voxel_y+i,:])
        axs[i//4,i%4].annotate('Omega: ' + str(round(omega[starting_voxel_x,starting_voxel_y+i],4)), 
                               xy=(0.2, 0), xycoords='axes fraction',
                               xytext=(-20, 20), textcoords='offset pixels',
                               horizontalalignment='left',
                               verticalalignment='bottom')
        title = 'Point Coordinates: [' + str(starting_voxel_x) + ',' + str(starting_voxel_y+i) + ']'
        axs[i//4,i%4].set_title(title)
    fig.set_size_inches(16, 9) # set figure's size manually
    plt.savefig('Fits of Curves Without An Offset', dpi=300)
    plt.show
    
    fig, axs=plt.subplots(4, 5, sharex = True, sharey=True)
    fig.suptitle("Images at flip angles",fontsize=18)
    for i in range(np.shape(slices)[2]):
        axs[i//5,i%5].imshow(slices[17:71,6:65,i].T)
        title = 'Flip Angle: ' + str(angles[i])
        axs[i//5,i%5].set_title(title)
        plt.subplots_adjust(top=0.933, bottom=0.16, left=0.3, right=0.7, hspace=0.214, wspace=0.0)
        plt.show

    #B1 Map
    fig, ax = plt.subplots()
    pos=ax.imshow(Omega_Normalised.T)
    fig.colorbar(pos, ax=ax)
    ax.set_title('Map of Normalised Omega Values Without Offset')
    fig.set_size_inches(9, 9) # set figure's size manually
    plt.savefig('Map of Normalised Omega Values Without Offset', dpi=300)
    plt.show()
    fig, ax = plt.subplots()
    pos=ax.imshow(B1Map.T)
    fig.colorbar(pos, ax=ax,ticks=[0, 10, 20,30,40,50,60,70,80,90,100],format=mticker.FixedFormatter(['0%','10%','20%','30%','40%','50%','60%','70%','80%','90%','100%']))
    ax.set_title('Map of Magnetic Field Values Without Offset')
    fig.set_size_inches(9, 9) # set figure's size manually
    plt.savefig('Map of B1 Values Without Offset', dpi=300)
    plt.show()

#Saving 
B1Map=nib.Nifti1Image(Omega_Normalised, nii.affine)
nib.save(B1Map, 'B1Map_38Deg_Nom.nii')