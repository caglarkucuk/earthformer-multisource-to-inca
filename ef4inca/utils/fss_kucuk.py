import torch
import numpy as np
import os
import pickle

from utils.fixedValues import NORM_DICT_TOTAL as normDict

from pysteps import verification
fss = verification.get_method("FSS")

def compute_fss(y, y_hat,
                tmp_name, save_dir,
                threshold_list, scale_list,
                device="cuda:0"):
    """Take tensors and save the scores per sample to the disk!"""
    ## Need to inverse the threshold transformation!
    threshold_list = np.power(10, threshold_list)
    if str(device) == "cuda:0":
        #### Denorm and detach while backtransforming!
        y = torch.pow(10,y).detach().cpu().numpy()
        y_hat = torch.pow(10,y_hat).detach().cpu().numpy()
    elif str(device) == "cpu":
        # Not sure how this would react!
        y = torch.pow(10,y).detach().numpy()
        y_hat = torch.pow(10,y_hat).detach().numpy()
    #### Loop over batch!
    assert y.shape[0] == 1, "Not implemented for batch size>1!"
    tmp_pred_event = y_hat[0,:,:,:,0]
    tmp_obs_event = y[0,:,:,:,0]
    usefulName = tmp_name[0].split('/')[-1].split('.')[0]
    #### Iterate over each time step and write on results!
    results = {(threshold, scale): np.zeros(tmp_pred_event.shape[0]) for threshold in threshold_list for scale in scale_list}
    for time_step in range(tmp_pred_event.shape[0]):
        pred = tmp_pred_event[time_step,...]
        obs = tmp_obs_event[time_step,...]
        # Iterate over each threshold and scale combination
        for threshold, scale in results.keys():
            # Here, 'fss' is your custom function
            fss_score = fss(pred, obs, thr=threshold, scale=scale)
            # Store the FSS score directly in the preallocated array for the current time_step
            results[(threshold, scale)][time_step] = fss_score
    # return results
    #### Save the results
    with open(os.path.join(save_dir, f'{usefulName}_fss_scores.pkl'), 'wb') as f:
        pickle.dump(results, f)