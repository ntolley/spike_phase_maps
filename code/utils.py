import numpy as np
import neo
import scipy.io as sio
import pandas as pd

def load_cgid_ev_explicit(fname):
    experiment_dict = sio.loadmat(fname)

    #Load variables from struct (struct indexing is unfortunately hideous)
    ev_ex = experiment_dict['ev_ex']
    timestamps = ev_ex['timestamps'][0][0][0]
    stage_go_cue = ev_ex['Stage'][0][0]['GoCue'][0][0][0]
    # stage_grasp_cue = ev_ex['Stage'][0][0]['GraspCue'][0][0][0]

    stage_stmv = ev_ex['Stage'][0][0]['StartMov'][0][0][0]
    stage_contact = ev_ex['Stage'][0][0]['Contact'][0][0][0]
    stage_object_present = ev_ex['Stage'][0][0]['ObjectPresent'][0][0][0]
    completion_completed = ev_ex['Completion'][0][0]['Completed'][0][0][0]
    object_tc = ev_ex['Object'][0][0]['TC'][0][0][0]
    grip_power = ev_ex['Grip'][0][0]['Power'][0][0][0]
    grip_precision = ev_ex['Grip'][0][0]['Precision'][0][0][0]
    grip_key = ev_ex['Grip'][0][0]['Key'][0][0][0]

    # Organize stage data into block format
    go = timestamps[np.logical_and(stage_go_cue, completion_completed)]
    contact = timestamps[np.logical_and(stage_contact, completion_completed)]
    obpres = timestamps[np.logical_and(stage_object_present, completion_completed)]
    stmv = timestamps[np.logical_and(stage_stmv, completion_completed)]
    # grasppres = timestamps[np.logical_and(stage_grasp_cue, completion_completed)]
    
    ob = object_tc[np.logical_and(stage_go_cue, completion_completed)]

    grip_power_block = grip_power[np.logical_and(stage_go_cue, completion_completed)]
    grip_precision_block = grip_precision[np.logical_and(stage_go_cue, completion_completed)] * 2
    grip_key_block = grip_key[np.logical_and(stage_go_cue, completion_completed)] * 3
    grip = grip_power_block + grip_precision_block + grip_key_block 

    return {'go': go, 'contact': contact, 'obpres': obpres, 'ob': ob, 'grip': grip,
            'stmv': stmv,}
            # 'grasppres': grasppres}