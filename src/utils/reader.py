import os
import numpy as np
import pandas as pd
import nibabel as nib

def _fid(id):
    '''Fills input id w/up to 5 leading 0s'''
    return str(id).zfill(5)

def _bid(id):
    '''Fills input id w/up to 5 leading 0s, prepends BraTS2021_ prefix'''
    return f"BraTS2021_{str(id).zfill(5)}"

def _sfid(fid):
    '''Strips all necessary leading 0s from input fid'''
    if fid == '00000':
        return 0
    else:
        return int(fid.lstrip('0'))

def load_mgmt_labels(mgmt_path=f'/home/{os.getlogin()}/brainworld/data/train_labels.csv'):
    return pd.read_csv(mgmt_path)

def get_overlap(seg_path=f'/home/{os.getlogin()}/brainworld/data/train'):
    '''Finds the overlap b/w segs files and mgmt label data'''
    df = load_mgmt_labels()
    mgmt_ids = [_fid(id) for id in df['BraTS21ID'].to_numpy()]
    seg_ids = [s.split('BraTS2021_')[-1] for s in os.listdir(seg_path)]
    overlap_fids = list(set(mgmt_ids).intersection(set(seg_ids)))
    return sorted([_sfid(fid) for fid in overlap_fids]) 

def find_crop(data):
    (x, y, z) = np.where(data > 0)
    return np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)

def crop_by(data, xmin, xmax, ymin, ymax, zmin, zmax, slices_step=1):
    return data[xmin:xmax, ymin:ymax, range(zmin, zmax, slices_step)]

def read_scan(id, modality='t2', train_dir=f'/home/{os.getlogin()}/brainworld/data/train'):
    assert modality in {'flair', 't1', 't1ce', 't2', 'seg'}
    scan = nib.load(f"{train_dir}/{_bid(id)}/{_bid(id)}_{modality}.nii.gz").get_fdata()
    if modality == 'seg': scan = scan.astype(np.uint8)
    if modality != 'seg': scan = scan/np.max(scan)
    return scan