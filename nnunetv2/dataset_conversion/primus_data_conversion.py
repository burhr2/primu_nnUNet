# convert the previous structure to nnUNet dataset structure
import os
import argparse
import shutil
import json

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import OrderedDict



def make_if_dont_exist(folder_path,overwrite=False):
    """
    creates a folder if it does not exists
    input: 
    folder_path : relative path of the folder which needs to be created
    over_write :(default: False) if True overwrite the existing folder 
    """
    if os.path.exists(folder_path):
        
        if not overwrite:
            print(f"{folder_path} exists.")
        else:
            print(f"{folder_path} overwritten")
            shutil.rmtree(folder_path)
            os.makedirs(folder_path)

    else:
      os.makedirs(folder_path)
      print(f"{folder_path} created!")
     

def main(args):
    
    task_description = args.task_desc
    dataset_path = args.datapath
    task_name = args.task_name
    task_id = args.task_id
    experiments_dir = args.output
    modalities = args.modality 

    path_dict = {
    "nnUNet_raw_data_base" : os.path.join(experiments_dir, "nnUNet_raw"), 
    "nnUNet_preprocessed" : os.path.join(experiments_dir, "nnUNet_preprocessed"), # 1 experiment: 1 epoch took 112s
    # "nnUNet_preprocessed" : os.path.join(base_dir, "nnUNet_preprocessed"), # 1 experiment: 1 epoch took 108s -> seems faster take this
    "RESULTS_FOLDER" : os.path.join(experiments_dir, "nnUNet_Results_Folder"),
    "RAW_DATA_PATH" : os.path.join(experiments_dir, "nnUNet_raw"), # This is used here only for convenience (not necessary for nnU-Net)!
    }       

    # Write paths to environment variables
    for env_var, path in path_dict.items():
        os.environ[env_var] = path 

    # Check whether all environment variables are set correct!
    for env_var, path in path_dict.items():
        if os.getenv(env_var) != path:
            print("Error:")
            print("Environment Variable {} is not set correctly!".format(env_var))
            print("Should be {}".format(path))
            print("Variable is {}".format(os.getenv(env_var)))
        make_if_dont_exist(path, overwrite=False)

    train_anat = os.path.join(dataset_path, r"train/anat/")
    train_seg = os.path.join(dataset_path, r"train/seg/")

    val_anat = os.path.join(dataset_path, r"val/anat/")
    val_seg = os.path.join(dataset_path, r"val/seg/")

    test_anat = os.path.join(dataset_path, r"test/anat/")
    test_seg = os.path.join(dataset_path, r"test/seg/")

    train_anat_dir = os.listdir(train_anat)
    train_seg_dir = os.listdir(train_seg)

    val_anat_dir = os.listdir(val_anat)
    val_seg_dir = os.listdir(val_seg)

    test_anat_dir = os.listdir(test_anat)
    test_seg_dir = os.listdir(test_seg)

    ## make sure we only have 0 or 1 in the masks
    for i, j in zip(train_seg_dir, train_anat_dir):
        id = os.path.join(train_seg, i)
        jd = os.path.join(train_anat, j)
        seg_nifti = nib.load(id)
        ant_nifti = nib.load(jd)

        seg_np = np.array(seg_nifti.dataobj)
        seg_np = np.where(seg_np > 0, 1, 0)
        ni_img = nib.Nifti1Image(seg_np, affine=ant_nifti.affine, dtype=np.float32)
        nib.save(ni_img, id)

    for i, j in zip(val_seg_dir, val_anat_dir):
        id = os.path.join(val_seg, i)
        jd = os.path.join(val_anat, j)

        seg_nifti = nib.load(id)
        anat_nifti = nib.load(jd)

        seg_np = np.array(seg_nifti.dataobj)
        seg_np = np.where(seg_np > 0, 1, 0)
        ni_img = nib.Nifti1Image(seg_np, affine=anat_nifti.affine, dtype=np.float32)
        ni_img.set_data_dtype(float)
        nib.save(ni_img, id)

    for i, j in zip(test_seg_dir, test_anat_dir):
        id = os.path.join(test_seg, i)
        jd = os.path.join(test_anat, j)

        # if ".csv" not in jd:
        seg_nifti = nib.load(id)
        anat_nifti = nib.load(jd)
        seg_np = np.array(seg_nifti.dataobj)
        seg_np = np.where(seg_np > 0, 1, 0)
        ni_img = nib.Nifti1Image(seg_np, affine=anat_nifti.affine , dtype=np.float32)
        ni_img.set_data_dtype(float)
        nib.save(ni_img, id)

    task_id_name = "Dataset" + str(task_id)+ "_" + str(task_name)
    processed_data_dir = os.path.join(experiments_dir, "nnUNet_raw/"+task_id_name)

    imagesTr = os.path.join(processed_data_dir,"imagesTr")
    labelsTr = os.path.join(processed_data_dir,"labelsTr")
    imagesTs = os.path.join(processed_data_dir,"imagesTs")
    labelsTs = os.path.join(processed_data_dir,"labelsTs")

    make_if_dont_exist(imagesTr)
    make_if_dont_exist(labelsTr)
    make_if_dont_exist(imagesTs)
    make_if_dont_exist(labelsTs)

    # copy files from old dir to new dir
    tr_anat = sorted(os.listdir(train_anat))
    tr_seg = sorted(os.listdir(train_seg))
    v_anat = sorted(os.listdir(val_anat))
    v_seg = sorted(os.listdir(val_seg))
    ts_anat = sorted(os.listdir(test_anat))
    ts_seg = sorted(os.listdir(test_seg))

    for file in tr_anat:
        id = os.path.join(train_anat, file)
        file_id = file.split(".")[0]
        # print(file_id)
        new_file_name = str(task_name)+"_"+file_id+"_0000.nii.gz"
        shutil.copy(id, os.path.join(imagesTr, new_file_name))

    for file in tr_seg:
        id = os.path.join(train_seg, file)
        file_id = file.split(".")[0]
        # print(file_id)
        new_file_name = str(task_name)+"_"+file_id+".nii.gz"
        shutil.copy(id, os.path.join(labelsTr, new_file_name))

    for file in ts_anat:
        id = os.path.join(test_anat, file)
        file_id = file.split(".")[0]
        # print(file_id)
        new_file_name = str(task_name)+"_"+file_id+"_0000.nii.gz"
        shutil.copy(id, os.path.join(imagesTs, new_file_name))

    for file in ts_seg:
        id = os.path.join(test_seg, file)
        file_id = file.split(".")[0]
        # print(file_id)
        new_file_name = str(task_name)+"_"+file_id+".nii.gz"
        shutil.copy(id, os.path.join(labelsTs, new_file_name))


    json_dict = OrderedDict()
    json_dict['name'] = task_name
    json_dict['description'] = task_description
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "Hussein et al"
    json_dict['licence'] = "primus"
    json_dict['release'] = "0.2"
    json_dict['modality'] = {}
    json_dict['labels'] = {
        "background":0,
        "lesion":1
    }

    for i, modality in enumerate(modalities):
            json_dict['modality'][str(i)] = modality

    json_dict['numTraining'] = len(tr_anat)
    json_dict['file_ending'] = ".nii.gz"
    json_dict['numTest'] = len(ts_anat)
    json_dict['training'] = [{'image': "./imagesTr/%s" %  (str(task_name)+"_"+i), "label": "./labelsTr/%s" %  (str(task_name)+"_"+i)} for i in os.listdir(train_anat)]
    json_dict['test'] = ["./imagesTs/%s" %  (str(task_name)+"_"+i) for i in os.listdir(test_anat)]


    with open(os.path.join(processed_data_dir, "dataset.json"), 'w') as f:
         json.dump(json_dict, f, indent=4, sort_keys=True)
    # return json_dict

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="prepare the data structure that nnUNet is expecting")
    parser.add_argument("--datapath", "-d", type=str, required=True, default="./data/")
    parser.add_argument("--modality", "-md", type=str, default=["MRI"])
    parser.add_argument("--output", "-o", type=str, required=True, default="./nnUNet_experiments")
    parser.add_argument("--task_name", "-tn", type=str, default="MSLesionSeg")
    parser.add_argument("--task_id", "-id", type=int, default=777)
    parser.add_argument("--task_desc", "-td", type=str, default="spinal cord ms lesion segmentation")

    args = parser.parse_args()
    main(args)

    
