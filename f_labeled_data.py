from pathlib import Path
from collections import OrderedDict

import pydicom
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import os
from imageio.v3 import imread
import highdicom as hd

import pandas

import pickle


def dicom2image(plan, width=None, level=None, norm=True):
    """
    Load windowed pixels from a DICOM Dataset
    
    See also: https://www.kaggle.com/code/wfwiggins203/eda-dicom-tags-windowing-head-cts/notebook
    """
    
    pixels = plan.pixel_array * plan.RescaleSlope + plan.RescaleIntercept

    if not width:
        width = plan.WindowWidth
        if not isinstance(width, pydicom.valuerep.DSfloat):
            width = width[0]
    if not level:
        level = plan.WindowCenter
        if not isinstance(level, pydicom.valuerep.DSfloat):
            level = level[0]

    lower = level - (width / 2)
    upper = level + (width / 2)
    img = np.clip(pixels, lower, upper)
    
    if norm:
        return (img - lower) / (upper - lower)
    else:
        return img

def segmentation2labels(segmentation, plan):
    """Returns an array of integer labels indicating segmentation locations.
    
    Inputs
        segmentation - a highdicom Segmentation object
        plan - a pydicom Dataset (the source multi-frame CT)
    
    Returns
        numpy array with 4th axis spanning different labels 0/1 encoded

    Only the NumberOfFrames attribute is used from plan.
    """
    sop_uid = segmentation.get_source_image_uids()[0][2]
    pixels = segmentation.get_pixels_by_source_frame(
        source_sop_instance_uid=sop_uid,
        # get a mask for all frames in the original dicom
        # i.e. get a mask for all slices of the CT
        source_frame_numbers=[x+1 for x in range(plan.NumberOfFrames)],
        # if a mask is not present for a frame, highdicom will raise an error,
        # as the segmentation plan contains no data for frames with no segmentations
        # setting this bool to true tells highdicom to assume the frames we request
        # exist, and assume that they are all 0s (i.e. no segmentation)
        assert_missing_frames_are_empty=True
    )
    return pixels

mask_map = OrderedDict([
    ['No segmentation', [0, 0, 0]],
    ['Acute C1', [255,0,0]],
    ['Acute C2', [255,127,0]],
    ['Acute C3', [255,212,0]],
    ['Acute C4', [237,185,185]],
    ['Acute C5', [143,35,35]],
    ['Acute C6', [143,106,35]],
    ['Acute C7', [79,143,35]],
    ['Chronic C1', [35,98,143]],
    ['Chronic C2', [107,35,143]],
    ['Chronic C3', [106,255,0]],
    ['Chronic C4', [0,234,255]],
    ['Chronic C5', [231,233,185]],
    ['Chronic C6', [185,215,237]],
    ['Chronic C7', [115,115,115]]
])

mask_labels = {i: k for i, k in enumerate(mask_map.keys())}
mask_rgb = {i: rgb for i, rgb in enumerate(mask_map.values())}

def mask2rgb(mask, slice_number, mask_map=mask_map):
    """Given an integer mask and slice, extract an RGB mask"""
    mask_slice = np.zeros([mask.shape[1], mask.shape[2], 3], dtype=np.uint8)

    # convert the 14 integers into rgb values
    for m, (label, rgb_value) in enumerate(mask_map.items()):
        if m == 0:
            continue

        # for slice s, see if there is a boolean True value for this segmentation
        # we use m-1 as the mask_map has 0 == No Segmentation, and this label is not
        # in the mask array
        idx = mask[slice_number, :, :, m-1] == 1
        if idx.any():
            # if there is, we have a positive segmentation here
            mask_slice[idx, :] = rgb_value
    
    return mask_slice


def slices2labels(exam_identifier):
    annotpath = 'gcs/annotations.csv'
    with open(annotpath, 'r') as af:
        annot_df = pandas.read_csv(af)

    pt_df = annot_df.loc[annot_df['exam_identifier'] == exam_identifier]

    slices2labels_dict = {}
    for index, row in pt_df.iterrows():
        slice_n = row['slice']
        acute_chronic = row['acute_chronic']
        if slice_n not in slices2labels_dict:
            if acute_chronic == 'acute':
                slices2labels_dict[slice_n] = np.uint8(1)
            else:
                slices2labels_dict[slice_n] = np.uint8(2)
    return slices2labels_dict


base_path = Path('gcs').resolve()
dicom_path = base_path / 'dicom'

study_ids = os.listdir(dicom_path)
# remove .dcm extension from study IDs
study_ids = [s.split('.')[0] for s in study_ids]

# getting NF slices from NF population
metapath = 'gcs/metadata.csv'
with open(metapath, 'r') as mf:
    meta_df = pandas.read_csv(mf)
f_patients = meta_df.iloc[:2,:]

f_patient_labels = []
for patient_id, acc_num in zip(f_patients['Patient_ID'], f_patients['Accession_Number']):
    study_id = patient_id + "_" + str(acc_num)
    filepath = dicom_path / study_id
    
    plan = pydicom.dcmread(dicom_path / f'{study_id}.dcm')
    volume = dicom2image(plan)

    s2l = slices2labels(study_id)
    
    for c, v in enumerate(volume):
        if c+1 in s2l:
            patient_label = [volume, s2l[c+1]]
        else:
            continue
            # patient_label = [volume, np.uint8(0)]
    
        f_patient_labels.append(patient_label)
pickle.dump(f_patient_labels, open('f_patient_data.pkl', 'wb'))
