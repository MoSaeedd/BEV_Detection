import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
from tqdm import tqdm
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.eval.detection.mAP_evaluation import get_average_precisions

import sys, getopt
from datetime import datetime
import pandas as pd
from transformation import  prepare_training_data_for_scene
from functools import partial
from multiprocessing import Pool
from train import train,predict,visualize_boxes,clean_up,load_groundtruth_boxes,get_pred_box3ds
from argparse import ArgumentParser

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
os.environ["OMP_NUM_THREADS"] = "1"


def visualize_lidar_of_sample(level5data,sample_token, axes_limit=80):
    sample = level5data.get("sample", sample_token)
    sample_lidar_token = sample["data"]["LIDAR_TOP"]
    level5data.render_sample_data(sample_lidar_token, axes_limit=axes_limit)
    



if __name__ == '__main__':

    parser = ArgumentParser(description='BEV Detection from Lidar & Camera')

    parser.add_argument('--dataset', dest='dataset_path',
                        const='Dataset', default='Dataset',
                        help='Dataset folder path')
    parser.add_argument('--json', dest='json_path'
                        ,default='/kaggle/input/3d-object-detection-for-autonomous-vehicles/train_data'
                        ,help='Path to folder containing training Json file')
    parser.add_argument('--train', dest='train', action='store_const',
                        const=True, default=False,
                        help='add this option for training')

    args = parser.parse_args()

    json_path = args.json_path
    dataset_path = args.dataset_path
    train = args.train

    # Our code will generate data, visualization and model checkpoints, they will be persisted to disk in this folder
    ARTIFACTS_FOLDER = "./artifacts"


    level5data = LyftDataset(data_path=dataset_path, json_path=json_path, verbose=True)
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]

    records = [(level5data.get('sample', record['first_sample_token'])['timestamp'], record) for record in level5data.scene]


    entries = []
    for start_time, record in sorted(records):
        start_time = level5data.get('sample', record['first_sample_token'])['timestamp'] / 1000000

        token = record['token']
        name = record['name']
        date = datetime.utcfromtimestamp(start_time)
        host = "-".join(record['name'].split("-")[:2])
        first_sample_token = record["first_sample_token"]

        entries.append((host, name, date, token, first_sample_token))
                
    df = pd.DataFrame(entries, columns=["host", "scene_name", "date", "scene_token", "first_sample_token"])
    host_count_df = df.groupby("host")['scene_token'].count()
    print(host_count_df)

    validation_hosts = ["host-a007", "host-a008", "host-a009"]
    validation_df = df[df["host"].isin(validation_hosts)]
    vi = validation_df.index
    train_df = df[~df.index.isin(vi)]
    print(len(train_df), len(validation_df), "train/validation split scene counts")

    # Some hyperparameters we'll need to define for the system
    voxel_size = (0.4, 0.4, 1.5)
    z_offset = -2.0
    bev_shape = (336, 336, 3)

    # We scale down each box so they are more separated when projected into our coarse voxel space.
    box_scale = 0.8

    train_data_folder = os.path.join(ARTIFACTS_FOLDER, "bev_train_data")
    validation_data_folder = os.path.join(ARTIFACTS_FOLDER, "./bev_validation_data")
    NUM_WORKERS = int(os.cpu_count()*1.5)

    if not os.path.isdir(ARTIFACTS_FOLDER) or not os.path.isdir(train_data_folder) or not os.path.isdir(validation_data_folder) :
        os.makedirs(ARTIFACTS_FOLDER, exist_ok=True)
        for df, data_folder in [(train_df, train_data_folder), (validation_df, validation_data_folder)]:
            print("Preparing data into {} using {} workers".format(data_folder, NUM_WORKERS))
            first_samples = df.first_sample_token.values

            os.makedirs(data_folder, exist_ok=True)
            
            process_func = partial(prepare_training_data_for_scene,
                                level5data=level5data,classes=classes,
                                output_folder=data_folder, bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset, box_scale=box_scale)

            pool = Pool(NUM_WORKERS)
            for _ in tqdm(pool.imap_unordered(process_func, first_samples), total=len(first_samples)):
                pass
            pool.close()
            del pool


    train_data_folder = os.path.join(ARTIFACTS_FOLDER, "bev_train_data")
    validation_data_folder = os.path.join(ARTIFACTS_FOLDER, "./bev_validation_data")

    # We weigh the loss for the 0 class lower to account for (some of) the big class imbalance.
    if not torch.cuda.is_available():
        print("Warning: No GPU Device found")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
    class_weights = class_weights.to(device)

    if train:
        train(train_data_folder,classes, ARTIFACTS_FOLDER)

    predictions_opened,predictions,detection_boxes,detection_classes,detection_scores,sample_tokens =predict(validation_data_folder,
                                                                            classes, 
                                                                            ARTIFACTS_FOLDER,
                                                                            class_weights)
    id = 0
    visualize_boxes(predictions_opened,detection_boxes,detection_scores,id)

    ## For Evaluation : gt_box3ds and pred_3d_boxes are used 
    gt_box3ds = load_groundtruth_boxes(level5data, sample_tokens)
    pred_3d_boxes = get_pred_box3ds(level5data,sample_tokens, detection_boxes, detection_scores, detection_classes
                    ,bev_shape, voxel_size, z_offset,box_scale)
    get_average_precisions(gt_box3ds, pred_3d_boxes, classes,0.5)
