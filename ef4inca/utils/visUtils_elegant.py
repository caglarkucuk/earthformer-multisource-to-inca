import os
from typing import List
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize, BoundaryNorm, LinearSegmentedColormap

from utils.fixedValues import NORM_DICT_TOTAL as normDict

import cartopy
import pyproj
import cartopy.crs as ccrs

rivers = cartopy.feature.NaturalEarthFeature(category='physical', name='rivers_lake_centerlines', scale='10m')
border = cartopy.feature.NaturalEarthFeature('cultural', 'admin_0_countries', '50m')

proj = pyproj.Proj("+init=epsg:31287 +no_defs")

NX = int((720000-20000)/1000)
NY = int((620000-220000)/1000)

y_coord = np.linspace(220000, 620000, NY + 1) + 1000 / 2.0
x_coord = np.linspace(20000, 720000, NX + 1) + 1000 / 2.0

x_grid, y_grid = np.meshgrid(x_coord, y_coord)

lon, lat = proj(x_grid, y_grid , inverse=True)

def get_cmap_0405(type):  # Adjusted to return a dictionary
    cmap_info = {}
    if type in ['CH5', 'CH6', 'CH7', 'CH9']: #== 'ir':  # SEVIRI channels
        colors1 = plt.cm.plasma(np.linspace(0., 1, 128))
        colors2 = plt.cm.gray_r(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        cmap_info['cmap'] = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
        vmin, vmax = 190, 290
    elif type == 'LGHT':
        # Generate the Reds color gradient
        reds_colors = plt.cm.copper_r(np.linspace(0, 1, 128))   
        # Prepend white
        colors = np.vstack(([1, 1, 1, 1], reds_colors))  # RGBA for white is [1, 1, 1, 1]
        # Create a new colormap from the extended color gradient
        cmap_info['cmap'] = mcolors.LinearSegmentedColormap.from_list('custom_red', colors)
        vmin, vmax = 0, 4
    elif type in ['INCA prep.', 'Radar prep.', 'CAPE', 'Target', 'EF4INCA']: # == 'precip':
        vmin, vmax = 0, 154
        intervals = [vmin, 0.2, 0.3, 0.6, 0.9, 1.7, 2.7, 5, 8.6, 15, 27.3, 50, 89.9, vmax]
        colors = ['#fffffe', '#0101ff', '#0153ff', '#00acfe', '#01feff',
                  '#8cff8c', '#fffe01', '#ff8d01', '#fe4300', '#f60100',
                  '#bc0030', '#ad01ac', '#ff00ff']
        cmap_info['cmap'] = LinearSegmentedColormap.from_list('custom_colormap', colors)
        cmap_info['norm'] = BoundaryNorm(intervals, cmap_info['cmap'].N)
        return cmap_info  # Early return as norm is already defined
    elif type == 'cape':
        vmin, vmax = 0, 300
        cmap_info['cmap'] = 'viridis'
    elif type == 'dem':
        vmin, vmax = 0, 4000
        #
        dem_colors = plt.cm.terrain(np.linspace(0.17, 1, 128))   
        # Prepend white
        colors = np.vstack(([1, 1, 1, 1], dem_colors))
        # Create a new colormap from the extended color gradient
        cmap_info['cmap'] = mcolors.LinearSegmentedColormap.from_list('dem_colors', colors)
    elif type == 'lat':
        vmin, vmax = 45, 50
        cmap_info['cmap'] = 'viridis'
    elif type == 'lon':
        vmin, vmax = 8, 18
        cmap_info['cmap'] = 'viridis'

    # For cases without custom norm, define a standard Normalize
    if 'norm' not in cmap_info:
        cmap_info['norm'] = Normalize(vmin=vmin, vmax=vmax)

    return cmap_info

def prepare_data2plot(in_seq, target_seq, pred_seq, normDict,
                      ):
    #### BREAK DOWN THE DATA!
    dem = in_seq[-7, :, :, -1]
    if in_seq.shape[0] == 25:
        cape = in_seq[-9:-7, :, :, -1] # 8th row for all aux
    else: # This can be generic since only 2h lead time uses two capes among the 3 lead time options!
        cape = in_seq[-8:-7, :, :, -1]
    radar = in_seq[:, :, :, -2] # 7th row
    inca_precip_x = in_seq[:, :, :, -3] # 6th row
    lght = in_seq[:, :, :, -4]  # 5th row
    seviri = in_seq[:, :, :, -8:-4] # This would still work even if there is only one sevir channel!
    ## Time to DEnormalize the data!
    seviri[:, :, :, -1] = (seviri[:, :, :, -1] * normDict['ch09']['std'] + normDict['ch09']['mean'])
    seviri[:, :, :, -2] = (seviri[:, :, :, -2] * normDict['ch07']['std'] + normDict['ch07']['mean'])
    seviri[:, :, :, -3] = (seviri[:, :, :, -3] * normDict['ch06']['std'] + normDict['ch06']['mean'])
    seviri[:, :, :, -4] = (seviri[:, :, :, -4] * normDict['ch05']['std'] + normDict['ch05']['mean'])
    lght = (lght * normDict['lght']['std'] + normDict['lght']['mean'])
    cape = (cape * normDict['inca_cape']['std'] + normDict['inca_cape']['mean'])
    dem = (dem * normDict['dem']['std'] + normDict['dem']['mean'])
    inca_precip_x = np.power(10, inca_precip_x)
    radar = np.power(10, radar)
    target_seq = np.power(10, target_seq)
    pred_seq = np.power(10, pred_seq)
    return seviri, lght, radar, inca_precip_x, cape, dem, target_seq, pred_seq
   
def plot_layer(ax, data, cmap_info, title=None, fs=10, show_cbar=False, cbar_label=None, row=None, col=None, plotting_ir=False, add_borders=True, add_rivers=False):
    """
    Enhanced function to plot a single layer with dynamic row and column handling.
    """
    if row is not None and col is not None:
        axis = ax[row, col]
    else:
        axis = ax

    # Ensure axis has a projection set, default to PlateCarree if not specified
    if axis.projection is None:
        axis.projection = ccrs.PlateCarree()

    im = axis.imshow(data, 
                     extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                     transform=ccrs.PlateCarree(),
                     **cmap_info)
    
    if title:
        axis.set_title(title, fontsize=fs)
    axis.xaxis.set_ticks([])
    axis.yaxis.set_ticks([])
    axis.set_aspect(700/400, adjustable='box')
    # Optionally add country borders
    if add_borders:    
        axis.add_feature(cartopy.feature.BORDERS, edgecolor='black')
    if add_rivers:
        # axis.add_feature(rivers, edgecolor='blue')
        axis.add_feature(cartopy.feature.RIVERS, edgecolor='blue')
    if show_cbar:
        # Create a new axis for the colorbar on the left of the plot
        cbar = plt.colorbar(im, ax=axis, location='left', orientation='vertical', fraction=0.035, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label, size=fs*.8)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
        if plotting_ir:
            cbar.set_ticks([200, 220, 240, 260, 280])
        cbar.ax.tick_params(labelsize=fs*.8)
    return im

def plot_past_regulars(ax, seviri, lght, inca_precip_x, radar, time_indices, time_points, cmap_dict, fs):
    """
    Plot SEVIRI channels dynamically based on the number of available bands.
    Channels are dropped in the order of 5, 6, and then 7, with channel 9 always present.
    """
    # Define the mapping for SEVIRI channels based on the input shape
    channel_mapping = {
        4: [5, 6, 7, 9]  # If 4 bands, they are channels 5, 6, 7, and 9
    }
    # Determine the available channels based on the number of SEVIRI bands
    num_bands = seviri.shape[-1]  # Assuming last dimension is channels
    channels = channel_mapping[num_bands]
    # Loop through the available channels and plot
    for row, channel in enumerate(channels):
        for col, time_idx in enumerate(time_indices):
            plot_layer(ax[row, col], seviri[time_idx, :, :, row], cmap_dict(f'CH{channel}'), show_cbar=(col==0), cbar_label=f'CH {channel}', plotting_ir=True, fs=fs)

    # Now loop through the rest of the regular data. Currently row=3 for full-case!
    for col, t_trans_idx in enumerate(time_indices):
        plot_layer(ax[row+1, col], lght[t_trans_idx, :, :], cmap_dict('LGHT'), show_cbar=(col==0), cbar_label='LGHT', fs=fs)
        plot_layer(ax[row+2, col], radar[t_trans_idx, :, :], cmap_dict('Radar prep.'), show_cbar=(col==0), cbar_label='Radar prep.', fs=fs)
        plot_layer(ax[row+3, col], inca_precip_x[t_trans_idx, :, :], cmap_dict('INCA prep.'), show_cbar=(col==0), cbar_label='INCA prep.', fs=fs)
        # Add the timestamps now!
        ax[0][col].set_title('{} Mins'.format(time_points[col]), fontsize=fs)
    return row + 4 # Return the row index for the next plot

def plot_future(ax, target_seq, pred_seq, time_indices, time_points, cmap_dict, row2work, fs):
    """
    Plot the future target and predicted sequences.
    """
    for col, t_trans_idx in enumerate(time_indices):
        plot_layer(ax[row2work, col], target_seq[t_trans_idx, :, :], cmap_dict('Target'), show_cbar=(col==0), cbar_label='Target', fs=fs)
        plot_layer(ax[row2work+1, col], pred_seq[t_trans_idx, :, :], cmap_dict('EF4INCA'), show_cbar=(col==0), cbar_label='EF4INCA', fs=fs)
        ax[row2work, col].set_title('{} Mins'.format(time_points[col]), fontsize=fs)

def plot_incaData(in_seq, target_seq, pred_seq, titleStr=None,
                               norm=normDict, 
                               figsize=(20, 15), # (w, h)
                               fs=10,
                               dpi=300, 
                               save_path=None,
                               savePdf=False,
                               ):
    """
    Refactored plotting function for INCA data.
    """
    # Data preparation
    seviri, lght, radar, inca_precip_x, cape, dem, \
        target_seq, pred_seq = prepare_data2plot(in_seq, target_seq, pred_seq, norm)
    # Set the colormap dictionary
    cmap_dict = lambda s: get_cmap_0405(s)
    nrows = seviri.shape[-1] + 6  # Additional rows for lght, inca_precip, radar, aux, target, and pred
    ncols = 7
    if pred_seq.shape[0] == 24:
        time_points_past = [-120, -90, -60, -30, -15, -5, 0]
        time_indices_past = [  0,   6,  12,  18,  21, 23, 24]
        #
        time_points_future = [5, 10, 15, 30, 60, 90, 120]
        time_indices_future = [0, 1,  2,  5, 11, 17, 23]
    elif pred_seq.shape[0] == 18:
        time_points_past =  [-90, -60, -30, -15, -10, -5, 0]
        time_indices_past = [  0,   6,  12,  15,  16, 17, 18]
        #
        time_points_future = [5, 10, 15, 30, 45, 60, 90]
        time_indices_future =[0,  1,  2,  5,  8, 11, 17]
    elif pred_seq.shape[0] == 12:
        time_points_past = [-60, -45, -30, -20, -10, -5, 0]
        time_indices_past =  [0,   3,   6,   8,  10, 11, 12]
        #
        time_points_future = [5, 10, 15, 20, 30, 45, 60]
        time_indices_future =[0,  1,  2,  3,  5,  8, 11]
    #
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, layout='constrained', subplot_kw={'projection': ccrs.PlateCarree()})
    ## First part is input data w/o aux!
    row2work = plot_past_regulars(ax, seviri, lght, inca_precip_x, radar, time_indices_past, time_points_past, cmap_dict, fs=fs)
    # Row of aux data will be plotted manually!
    if seviri.shape[0] == 25:
        plot_layer(ax[row2work, 0], cape[0,:,:], cmap_dict('cape'), show_cbar=True, cbar_label='CAPE', title='INCA CAPE t=-2 H.', fs=fs)
        plot_layer(ax[row2work, 1], cape[1,:,:], cmap_dict('cape'), title='CAPE t=-1 H.', fs=fs)
        col4aux = 2
    else: # This can be generic since only 2h lead time uses two capes among the 3 lead time options!
        plot_layer(ax[row2work, 0], cape[0,:,:], cmap_dict('cape'), show_cbar=True, cbar_label='CAPE', title='t=-1 H.', fs=fs)
        col4aux = 1
    # Turn for DEM #### Don't plot lat and lon!, LAT, and LON
    plot_layer(ax[row2work, col4aux], dem, cmap_dict('dem'), title='DEM', fs=fs) # , show_cbar=True
    # Rest is empty!
    for i in range(col4aux+1, 7):
        ax[row2work, i].axis('off')    
    #### Time to go to the future!
    row2work += 1
    plot_future(ax, target_seq, pred_seq, time_indices_future, time_points_future, cmap_dict, row2work, fs)
    #
    if titleStr:
        fig.suptitle(titleStr, fontsize=fs*1.5)
    #
    if save_path:
        if savePdf:
            fig.savefig(save_path.replace('.png', '.pdf'), dpi=dpi, bbox_inches='tight')
        else:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def save_example_vis_results(
        save_dir, save_prefix, in_seq, target_seq, pred_seq, label,
        fs=24, norm=None, title=None,
        savePdf=False
        ):
    in_seq = in_seq[0,:,:,:,:].astype(np.float32)
    target_seq = target_seq[0,:,:,:,0].astype(np.float32)
    pred_seq = pred_seq[0,:,:,:,0].astype(np.float32)
    timestamp = title[0].split('_')[-1][:-3]
    fig_path = save_dir +'/'+ save_prefix + '_' + timestamp + '.png'

    if savePdf: # We also need to pimp up the title in this case!
        timestamp = f'{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[8:10]}:{timestamp[10:12]}'
        title = f'Sample at t = {timestamp}'
        fig_path = fig_path.replace('.png', '.pdf')
    else:
        title = title[0].split('/')[-1][:-3]

    plot_incaData(in_seq=in_seq, target_seq=target_seq, pred_seq=pred_seq, 
        titleStr=title, 
        save_path=fig_path,
        savePdf=savePdf, 
        fs=fs, 
        figsize=(15, 16)
        )