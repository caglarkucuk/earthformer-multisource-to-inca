"""Run pysteps as a baseline for predictions!
Mostly following the deterministic prediction section of 
https://pysteps.readthedocs.io/en/stable/auto_examples/plot_steps_nowcast.html

Base EF4INCA environment doesn't have pysteps installed in it. You need another environment with 
pysteps to run this script!

"""
import os
import numpy as np
import h5py

from tqdm.autonotebook import tqdm

from pysteps.motion.lucaskanade import dense_lucaskanade
from pysteps import nowcasts
nowcast_method = nowcasts.get_method("sprog")

# Set nowcast parameters
n_leadtimes = 18
seed = 24

## Set dirs and files
dir2read = '/PATH/TO/TEST/DATASET/'
dir2write = '/PATH/TO/WRITE/PREDICTIONS/' # Careful with disk space!

def read_file(file, verbose=False, 
              nt_start=5, nt_future=18, nt_past=19 # Hardcoded to reflect the original EF4INCA config
              ):
    '''Quick and pragmatic adaptation of the original file reader available 
    in the kucukINCAdataset class!'''
    ## Read the file in!
    eventFile = h5py.File(file, 'r')
    # Read the data with required indexing
    inca_precip_x = eventFile['inca_precip'][nt_start:(nt_start+nt_past),:,:]
    inca_precip_y = eventFile['inca_precip'][(nt_start+nt_past):(nt_start+nt_past+nt_future),:,:]
    if verbose:
            print(f'inca_precip_x shape: {inca_precip_x.shape}, inca_precip_y shape: {inca_precip_y.shape}')
    return inca_precip_x, inca_precip_y

def logTransform(x):
    """Log transformation of the data as done in the EF model!
    Set anything less than 0.1 to 0.02 and then take the log10 of the data!"""
    x[x < 0.1] = 0.02
    x = np.log10(x)
    return x

def backTransform(x):
    """Back transformation of the data as done in the EF model!
    Take the 10^x of the data!
    IN ADDITION, SET nan values to 0"""
    x = np.power(10, x)
    x[np.isnan(x)] = 0
    return x

## Get the files to work on!
files2surf = os.listdir(dir2read)
# Add the dir if extesnion is h5!
files2surf = [file for file in files2surf if file.endswith('.h5')]
files2surf.sort()

kk=0
for file in tqdm(files2surf):
    # Read the data
    obs_x, obs_y = read_file(dir2read+file) # , verbose=True
    # Log transform the data
    obs_x = logTransform(obs_x)
    # Compute the motion vector!
    V = dense_lucaskanade(obs_x)
    #### Compute the nowcast
    # Need to wrap the nowcast for cases with complete 0 values!
    try:
        # The S-PROG nowcast
        R_f = nowcast_method(
            obs_x[-3:, :, :],
            V,
            n_leadtimes,
            n_cascade_levels=6,
            R_thr=-1.0, # This is np.log10(0.1) !!!
        )
    except ValueError as e:
        if "x contains non-finite values" in str(e):
            print("Non-finite values detected in input data. Setting R_f to zeros.")
            # Just set this to nan and it'll be 'backtransformed' to 0 in the next step!
            R_f = np.empty(obs_y.shape)
            R_f[:] = np.nan
        else:
            raise  # Re-raise the exception if it's not the specific ValueError we're handling

    # Backtransform the nowcast and save
    R_f = backTransform(R_f)
    # Save the nowcast
    with h5py.File(dir2write + file, 'w') as f:
        f.create_dataset('pysteps_pred', data=R_f, compression='gzip', compression_opts=1)
        f.create_dataset('target', data=obs_y, compression='gzip', compression_opts=1) # This doesn't have to be saved
