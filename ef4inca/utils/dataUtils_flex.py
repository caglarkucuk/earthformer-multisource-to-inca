import glob
import random
import numpy as np
import h5py
from datetime import datetime, timedelta
import pandas as pd
import os

import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from scipy.ndimage import zoom # Upsampling of SEVIRI!

from utils.fixedValues import NORM_DICT_TOTAL as normDict

# Write a function that returns a 2d matrix of the lightning counts for a given 5-minute period!
def lght2rast(df4rast, event_raster, time_idx):
    """event_raster should have a shape of t, 400, 700"""
    ## Introduce the metadata!
    xpixelsize = 1000.      # INCA grid mesh size west-east [m]
    ypixelsize = 1000.      # INCA grid mesh size south-north [m]
    x1 = 20000     # INCA grid western boundary index [m]
    y2 = 620000    # INCA grid northern boundary index [m]
    # Calculate the pixel coordinates in the raster for each event
    df4rast = df4rast.copy()
    df4rast['pixel_x'] = ((df4rast['x_proj'] - x1) / xpixelsize).astype(int)
    df4rast['pixel_y'] = ((y2 - df4rast['y_proj']) / ypixelsize).astype(int)
    # Make sure that the pixel coordinates are within the boundaries!
    df4rast = df4rast[(df4rast['pixel_x'] >= 0) & (df4rast['pixel_x'] < 700) & (df4rast['pixel_y'] >= 0) & (df4rast['pixel_y'] < 400)]
    # Count the events per pixel
    event_counts = df4rast.groupby(['pixel_x', 'pixel_y']).size().reset_index(name='counts')
    # Update the raster with event counts
    for index, row in event_counts.iterrows():
        event_raster[time_idx, row['pixel_y'], row['pixel_x']] = row['counts']
    return event_raster

class kucukINCAdataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, 
                img_list, 
                nt_start=0, 
                nt_future=24, # Set this to 0 if you want to load the whole sequence as past data!
        ): # 
        self.nt_start = nt_start
        self.nt_future = nt_future
        if nt_future == 24:
            self.nt_past = 25
        elif nt_future == 12:
            self.nt_past = 13
        elif nt_future == 18:
            self.nt_past = 19
        ##
        self.img_list = img_list
        print(f'num ims: {len(img_list)}')
        ##
        self.h_zoom_seviri = 400 / 86  # 400 and 700 are INCA shape,
        self.w_zoom_seviri = 700 / 150 # while 86 and 150 are SEVIRI shape for INCA domain!
        ##
        dat_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        self.dem = h5py.File(dat_dir+'/data/Aux/matched_eu_dem.h5','r')['raster_data'][:,:]
        self.lat = h5py.File(dat_dir+'/data/Aux/INCA_coords.h5','r')['lat'][1:, :-1]
        self.lon = h5py.File(dat_dir+'/data/Aux/INCA_coords.h5','r')['lon'][1:, :-1]
        #
        self.dem = (self.dem - normDict['dem']['mean']) / normDict['dem']['std']
        self.lat = (self.lat - normDict['lat']['mean']) / normDict['lat']['std']
        self.lon = (self.lon - normDict['lon']['mean']) / normDict['lon']['std']

    def __len__(self):
        # Denotes the number of batches per epoch
        return len(self.img_list)

    def __getitem__(self, index):
        # Generate one sample of data
        file_name = self.img_list[index]
        eventFile = h5py.File(file_name, 'r')
        # Get the timestamp from file_name!
        timestamp = file_name.split('_')[-1][:-3]
        # Make it a time object!
        timestamp = datetime.strptime(timestamp, '%Y%m%d%H%M')
        # Get the day of year and hour of day!        
        time_of_day = timestamp.hour + timestamp.minute / 60 
        # Set the start time of the first observation for lght!
        startTime_lght = timestamp - timedelta(minutes=130)
        ## Work on seviri!
        # Read the data
        seviri = eventFile['seviri'][self.nt_start:(self.nt_start+self.nt_past),:,:,:] # (t, h, w, c)  
        # Match the resolution with target!
        seviri = zoom(seviri, (1, self.h_zoom_seviri, self.w_zoom_seviri, 1), order=1) # bilinear interpolation
        ## Work on radar!
        # Read the data
        radar = eventFile['radar'][self.nt_start:(self.nt_start+self.nt_past),:,:] # (t, h, w)
        ## Work on lght!
        # Read the data
        lght_df = pd.DataFrame(eventFile['lght'][:,:], columns=['event_time', 'x_proj',	'y_proj'])
        # Convert the 'timestamp' column to datetime
        lght_df['event_time'] = pd.to_datetime(lght_df['event_time'], unit='s')
        # Rasterize the lght data!
        lght = np.zeros((self.nt_past, 400, 700))
        for i in range(self.nt_past):
            lght = lght2rast(lght_df[(lght_df['event_time'] > startTime_lght) & (lght_df['event_time'] <= (startTime_lght+timedelta(minutes=5)))], 
                             lght, i)
            startTime_lght += timedelta(minutes=5)
        ## Work on inca precip!
        # Read the data
        inca_precip_x = eventFile['inca_precip'][self.nt_start:(self.nt_start+self.nt_past),:,:]
        inca_precip_y = eventFile['inca_precip'][(self.nt_start+self.nt_past):(self.nt_start+self.nt_past+self.nt_future),:,:]
        
        ## Time to normalize the data!
        seviri[:, :, :, 0] = (seviri[:, :, :, 0] - normDict['ch05']['mean']) / normDict['ch05']['std']
        seviri[:, :, :, 1] = (seviri[:, :, :, 1] - normDict['ch06']['mean']) / normDict['ch06']['std']
        seviri[:, :, :, 2] = (seviri[:, :, :, 2] - normDict['ch07']['mean']) / normDict['ch07']['std']
        seviri[:, :, :, 3] = (seviri[:, :, :, 3] - normDict['ch09']['mean']) / normDict['ch09']['std']
        lght = (lght - normDict['lght']['mean']) / normDict['lght']['std']
        
        # Normalization of precip 
        """OK, let's stick to LDCast approach, add the thr. from 0.1 and shift 
        everytthing below that to 0.02 before log10 transforming precip data!"""
        inca_precip_x[inca_precip_x < 0.1] = 0.02
        inca_precip_y[inca_precip_y < 0.1] = 0.02
        radar[radar < 0.1] = 0.02
        inca_precip_x = np.log10(inca_precip_x)
        inca_precip_y = np.log10(inca_precip_y)
        radar = np.log10(radar)
        
        #### Work on aux data! ####
        ## Work on inca cape!
        # Read the data
        if self.nt_past == 25:
            inca_cape = eventFile['inca_cape'][:2,:,:] #### ONLY TWO ESTIMATES ARE USED!!!!
        elif self.nt_past == 13:
            inca_cape = eventFile['inca_cape'][1:2,:,:] #### ONLY last estimate is used!
        elif self.nt_past == 19:
            inca_cape = eventFile['inca_cape'][1:2,:,:] #### ONLY last estimate is used here too!
        inca_cape = (inca_cape - normDict['inca_cape']['mean']) / normDict['inca_cape']['std']
        # Add DoY and HoD as sine waves!
        doy_sin = np.ones((400, 700)) * np.sin(timestamp.timetuple().tm_yday * 2 * np.pi / 365.2425)
        doy_cos = np.ones((400, 700)) * np.cos(timestamp.timetuple().tm_yday * 2 * np.pi / 365.2425)
        hod_sin = np.ones((400, 700)) * np.sin(time_of_day * 2 * np.pi / 24)
        hod_cos = np.ones((400, 700)) * np.cos(time_of_day * 2 * np.pi / 24)
        # Get the number of zero layers in the first 'time steps' of aux data!
        missing_steps = self.nt_past - inca_cape.shape[0] - 1 - 2 - 2 - 2 # 1 for dem, 2 for coords, 2 for sine waves of time, 2 for cos waves of time
        # Combine the aux 'band'
        combined_aux = np.concatenate((np.zeros((missing_steps, 400, 700)), inca_cape, 
                                       self.dem[np.newaxis, :, :], 
                                       self.lat[np.newaxis, :, :], self.lon[np.newaxis, :, :], 
                                       doy_sin[np.newaxis, :, :], doy_cos[np.newaxis, :, :], 
                                       hod_sin[np.newaxis, :, :], hod_cos[np.newaxis, :, ]), axis=0)
        ## Finally, concatenate all of the data!
        sample_past = np.concatenate((seviri, lght[:, :, :, np.newaxis], inca_precip_x[:, :, :, np.newaxis], 
                                      radar[:, :, :, np.newaxis], combined_aux[:, :, :, np.newaxis]), axis=-1)
        sample_future = inca_precip_y[:, :, :, np.newaxis]
        ## Push them to torch and return a sample
        sample_past = torch.from_numpy(sample_past.astype(np.float32))
        sample_future = torch.from_numpy(sample_future.astype(np.float32))
        sample = {'sample_past': sample_past, 'sample_future': sample_future, 'name': file_name}
        return sample

class kucukINCAdataModule(pl.LightningDataModule):
    def __init__(self, 
                params, 
                data_dir,
                stage='train', 
                ):
        super().__init__() 
        self.params = params

        ## Let's define the fileList here!
        trainFilesTotal = glob.glob(data_dir+'train/*.h5') 
        valFilesTotal = glob.glob(data_dir+'val/*.h5')
        random.seed(0)
        random.shuffle(trainFilesTotal)  # shuffle the file list randomly

        self.testFiles = glob.glob(data_dir+'test/*.h5')

        self.trainFiles = trainFilesTotal
        self.valFiles = valFilesTotal

    def setup(self, stage=None):
        if self.params['out_len'] == 24:
            nt_start = 0
            nt_future = 24
        elif self.params['out_len'] == 12:
            nt_start = 11
            nt_future = 12
        elif self.params['out_len'] == 18:
            nt_start = 5
            nt_future = 18
        if stage == 'train' or stage is None:
            self.train_dataset = kucukINCAdataset(img_list=self.trainFiles, 
                                                  nt_start=nt_start, nt_future=nt_future,
                                                  )
            self.val_dataset = kucukINCAdataset(img_list=self.valFiles,
                                                nt_start=nt_start, nt_future=nt_future,
                                                )
        if stage == 'test':
            self.test_dataset = kucukINCAdataset(img_list=self.testFiles, 
                                                 nt_start=nt_start, nt_future=nt_future,
                                                 )
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params['BATCH_SIZE'], shuffle=True, num_workers=self.params['NUM_WORKERS'])

    def val_dataloader(self):
        # No shuffling!
        return DataLoader(self.val_dataset, batch_size=self.params['BATCH_SIZE'], shuffle=False, num_workers=self.params['NUM_WORKERS'])

    def test_dataloader(self):
        # Batch size is 1 for testing! No shuffling!
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.params['NUM_WORKERS'])
    
    #### These properties are added from the original repository!
    @property
    def num_train_samples(self):
        return len(self.train_dataset)

    @property
    def num_val_samples(self):
        return len(self.val_dataset)

    @property
    def num_test_samples(self):
        return len(self.test_dataset)
