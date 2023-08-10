
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
from model import get_unet_model
import os 
import cv2 
from tqdm import tqdm
import glob
import shutil
import torch.nn.functional as F
from dataset import BEVImageDataset
from lyft_dataset_sdk.eval.detection.mAP_evaluation import Box3D
from lyft_dataset_sdk.utils.data_classes import  Quaternion
from lyft_dataset_sdk.utils.geometry_utils import transform_matrix
from transformation import create_transformation_matrix_to_voxel_space, transform_points
from scipy.spatial.transform import Rotation as R



def load_groundtruth_boxes(level5data, sample_tokens):
    gt_box3ds = []

    # Load annotations and filter predictions and annotations.
    for sample_token in tqdm(sample_tokens):

        sample = level5data.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])
        
        for sample_annotation_token in sample_annotation_tokens:
            sample_annotation = level5data.get('sample_annotation', sample_annotation_token)
            sample_annotation_translation = sample_annotation['translation']
            
            class_name = sample_annotation['category_name']
            
            box3d = Box3D(
                sample_token=sample_token,
                translation=sample_annotation_translation,
                size=sample_annotation['size'],
                rotation=sample_annotation['rotation'],
                name=class_name
            )
            gt_box3ds.append(box3d)
            
    return gt_box3ds

def visualize_predictions(input_image, prediction, target, n_images=2, apply_softmax=True):
    """
    Takes as input 3 PyTorch tensors, plots the input image, predictions and targets.
    """
    # Only select the first n images
    prediction = prediction[:n_images]
    target = target[:n_images]
    input_image = input_image[:n_images]

    prediction = prediction.detach().cpu().numpy()
    if apply_softmax:
        prediction = scipy.special.softmax(prediction, axis=1)
    class_one_preds = np.hstack(1-prediction[:,0])

    target = np.hstack(target.detach().cpu().numpy())

    class_rgb = np.repeat(class_one_preds[..., None], 3, axis=2)
    class_rgb[...,2] = 0
    class_rgb[...,1] = target

    
    input_im = np.hstack(input_image.cpu().numpy().transpose(0,2,3,1))
    
    if input_im.shape[2] == 3:
        input_im_grayscale = np.repeat(input_im.mean(axis=2)[..., None], 3, axis=2)
        overlayed_im = (input_im_grayscale*0.6 + class_rgb*0.7).clip(0,1)
    else:
        input_map = input_im[...,3:]
        overlayed_im = (input_map*0.6 + class_rgb*0.7).clip(0,1)

    thresholded_pred = np.repeat(class_one_preds[..., None] > 0.5, 3, axis=2)

    fig = plt.figure(figsize=(12,26))
    plot_im = np.vstack([class_rgb, input_im[...,:3], overlayed_im, thresholded_pred]).clip(0,1).astype(np.float32)
    plt.imshow(plot_im)
    plt.axis("off")
    plt.show()

def train(train_data_folder,classes, ARTIFACTS_FOLDER):

    input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.png")))
    target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))

    train_dataset = BEVImageDataset(input_filepaths, target_filepaths)
        
    # im, target, sample_token = train_dataset[1]
    # im = im.numpy()
    # target = target.numpy()

    # plt.figure(figsize=(16,8))

    # target_as_rgb = np.repeat(target[...,None], 3, 2)
    # # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
    # plt.imshow(np.hstack((im.transpose(1,2,0)[...,:3], target_as_rgb)))
    # plt.title(sample_token)
    # plt.show()



    # We weigh the loss for the 0 class lower to account for (some of) the big class imbalance.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
    class_weights = class_weights.to(device)


    batch_size = 8
    epochs = 15 # Note: We may be able to train for longer and expect better results, the reason this number is low is to keep the runtime short.

    model = get_unet_model(num_output_classes=len(classes)+1)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=os.cpu_count()*2)

    all_losses = []

    for epoch in range(1, epochs+1):
        print("Epoch", epoch)
        
        epoch_losses = []
        progress_bar = tqdm(dataloader)
        
        for ii, (X, target, sample_ids) in enumerate(progress_bar):
            X = X.to(device)  # [N, 3, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model(X)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            epoch_losses.append(loss.detach().cpu().numpy())

            if ii == 0:
                visualize_predictions(X, prediction, target)
        
        print("Loss:", np.mean(epoch_losses))
        all_losses.extend(epoch_losses)
        
        checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch)
        checkpoint_filepath = os.path.join(ARTIFACTS_FOLDER, checkpoint_filename)
        torch.save(model.state_dict(), checkpoint_filepath)
        
    plt.figure(figsize=(12,12))
    plt.plot(all_losses, alpha=0.75)
    plt.show()

def predict(validation_data_folder,classes, ARTIFACTS_FOLDER,class_weights):
    input_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_input.png")))
    target_filepaths = sorted(glob.glob(os.path.join(validation_data_folder, "*_target.png")))

    batch_size=16
    validation_dataset = BEVImageDataset(input_filepaths, target_filepaths)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size, shuffle=False, num_workers=os.cpu_count())


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_unet_model(num_output_classes=1+len(classes))
    model = model.to(device)

    epoch_to_load=15
    checkpoint_filename = "unet_checkpoint_epoch_{}.pth".format(epoch_to_load)
    checkpoint_filepath = os.path.join(ARTIFACTS_FOLDER, checkpoint_filename)
    model.load_state_dict(torch.load(checkpoint_filepath))


    progress_bar = tqdm(validation_dataloader)

    targets = np.zeros((len(target_filepaths), 336, 336), dtype=np.uint8)

    # We quantize to uint8 here to conserve memory. We're allocating >20GB of memory otherwise.
    predictions = np.zeros((len(target_filepaths), 1+len(classes), 336, 336), dtype=np.uint8)

    sample_tokens = []
    all_losses = []

    with torch.no_grad():
        model.eval()
        for ii, (X, target, batch_sample_tokens) in enumerate(progress_bar):

            offset = ii*batch_size
            targets[offset:offset+batch_size] = target.numpy()
            sample_tokens.extend(batch_sample_tokens)
            
            X = X.to(device)  # [N, 1, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            prediction = model(X)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)
            all_losses.append(loss.detach().cpu().numpy())
            
            prediction = F.softmax(prediction, dim=1)
            
            prediction_cpu = prediction.cpu().numpy()
            predictions[offset:offset+batch_size] = np.round(prediction_cpu*255).astype(np.uint8)
            
            # Visualize the first prediction
            if ii == 0:
                visualize_predictions(X, prediction, target, apply_softmax=False)
                
    print("Mean loss:", np.mean(all_losses))



    # Get probabilities for non-background
    predictions_non_class0 = 255 - predictions[:,0]

    # Arbitrary threshold in our system to create a binary image to fit boxes around.
    background_threshold = 255//2
    for i in range(3):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
        axes[0].imshow(predictions_non_class0[i])
        axes[0].set_title("predictions")
        axes[1].imshow(predictions_non_class0[i] > background_threshold)
        axes[1].set_title("thresholded predictions")
        axes[2].imshow((targets[i] > 0).astype(np.uint8), interpolation="nearest")
        axes[2].set_title("targets")
        fig.tight_layout()
        fig.show()


    # We perform an opening morphological operation to filter tiny detections
    # Note that this may be problematic for classes that are inherently small (e.g. pedestrians)..
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    predictions_opened = np.zeros((predictions_non_class0.shape), dtype=np.uint8)

    for i, p in enumerate(tqdm(predictions_non_class0)):
        thresholded_p = (p > background_threshold).astype(np.uint8)
        predictions_opened[i] = cv2.morphologyEx(thresholded_p, cv2.MORPH_OPEN, kernel)

    plt.figure(figsize=(12,12))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    axes[0].imshow(predictions_non_class0[0] > 255//2)
    axes[0].set_title("thresholded prediction")
    axes[1].imshow(predictions_opened[0])
    axes[1].set_title("opened thresholded prediction")
    fig.show()

    # Sanity check: let's count the amount of connected components in an image
    labels, n = scipy.ndimage.label(predictions_opened[0])
    plt.imshow(labels, cmap="tab20b")
    plt.xlabel("N predictions: {}".format(n))
    plt.show()
    detection_boxes = []
    detection_scores = []
    detection_classes = []

    for i in tqdm(range(len(predictions))):
        prediction_opened = predictions_opened[i]
        probability_non_class0 = predictions_non_class0[i]
        class_probability = predictions[i]

        sample_boxes = []
        sample_detection_scores = []
        sample_detection_classes = []
        
        contours, hierarchy = cv2.findContours(prediction_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            
            # Let's take the center pixel value as the confidence value
            box_center_index = np.int0(np.mean(box, axis=0))
            
            for class_index in range(len(classes)):
                box_center_value = class_probability[class_index+1, box_center_index[1], box_center_index[0]]
                
                # Let's remove candidates with very low probability
                if box_center_value < 0.01:
                    continue
                
                box_center_class = classes[class_index]

                box_detection_score = box_center_value
                sample_detection_classes.append(box_center_class)
                sample_detection_scores.append(box_detection_score)
                sample_boxes.append(box)
            
        
        detection_boxes.append(np.array(sample_boxes))
        detection_scores.append(sample_detection_scores)
        detection_classes.append(sample_detection_classes)
        
    print("Total amount of boxes:", np.sum([len(x) for x in detection_boxes]))
    return prediction_opened,predictions,detection_boxes,detection_classes,detection_scores,sample_tokens

# Visualize the boxes 
def visualize_boxes(predictions_opened,detection_boxes,detection_scores,id):
    t = np.zeros_like(predictions_opened[id])
    for sample_boxes in detection_boxes[id]:
        box_pix = np.int0(sample_boxes)
        cv2.drawContours(t,[box_pix],0,(255),2)
    plt.imshow(t)
    plt.show()

    # Visualize their probabilities
    plt.hist(detection_scores[0], bins=20)
    plt.xlabel("Detection Score")
    plt.ylabel("Count")
    plt.show()

# gt_box3ds = load_groundtruth_boxes(level5data, sample_tokens)

def get_pred_box3ds(level5data,sample_tokens, detection_boxes, detection_scores, detection_classes
                    ,bev_shape, voxel_size, z_offset,box_scale):
    pred_box3ds = []

    # This could use some refactoring..
    for (sample_token, sample_boxes, sample_detection_scores, sample_detection_class) in tqdm(zip(sample_tokens, detection_boxes, detection_scores, detection_classes), total=len(sample_tokens)):
        sample_boxes = sample_boxes.reshape(-1, 2) # (N, 4, 2) -> (N*4, 2)
        sample_boxes = sample_boxes.transpose(1,0) # (N*4, 2) -> (2, N*4)

        # Add Z dimension
        sample_boxes = np.vstack((sample_boxes, np.zeros(sample_boxes.shape[1]),)) # (2, N*4) -> (3, N*4)

        sample = level5data.get("sample", sample_token)
        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = level5data.get("sample_data", sample_lidar_token)
        lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
        ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
        ego_translation = np.array(ego_pose['translation'])

        global_from_car = transform_matrix(ego_pose['translation'],
                                        Quaternion(ego_pose['rotation']), inverse=False)

        car_from_voxel = np.linalg.inv(create_transformation_matrix_to_voxel_space(bev_shape, voxel_size, (0, 0, z_offset)))


        global_from_voxel = np.dot(global_from_car, car_from_voxel)
        sample_boxes = transform_points(sample_boxes, global_from_voxel)

        # We don't know at where the boxes are in the scene on the z-axis (up-down), let's assume all of them are at
        # the same height as the ego vehicle.
        sample_boxes[2,:] = ego_pose["translation"][2]


        # (3, N*4) -> (N, 4, 3)
        sample_boxes = sample_boxes.transpose(1,0).reshape(-1, 4, 3)


        # We don't know the height of our boxes, let's assume every object is the same height.
        box_height = 1.75

        # Note: Each of these boxes describes the ground corners of a 3D box.
        # To get the center of the box in 3D, we'll have to add half the height to it.
        sample_boxes_centers = sample_boxes.mean(axis=1)
        sample_boxes_centers[:,2] += box_height/2

        # Width and height is arbitrary - we don't know what way the vehicles are pointing from our prediction segmentation
        # It doesn't matter for evaluation, so no need to worry about that here.
        # Note: We scaled our targets to be 0.8 the actual size, we need to adjust for that
        sample_lengths = np.linalg.norm(sample_boxes[:,0,:] - sample_boxes[:,1,:], axis=1) * 1/box_scale
        sample_widths = np.linalg.norm(sample_boxes[:,1,:] - sample_boxes[:,2,:], axis=1) * 1/box_scale
        
        sample_boxes_dimensions = np.zeros_like(sample_boxes_centers) 
        sample_boxes_dimensions[:,0] = sample_widths
        sample_boxes_dimensions[:,1] = sample_lengths
        sample_boxes_dimensions[:,2] = box_height

        for i in range(len(sample_boxes)):
            translation = sample_boxes_centers[i]
            size = sample_boxes_dimensions[i]
            class_name = sample_detection_class[i]
            ego_distance = float(np.linalg.norm(ego_translation - translation))
        
            
            # Determine the rotation of the box
            v = (sample_boxes[i,0] - sample_boxes[i,1])
            v /= np.linalg.norm(v)
            r = R.from_dcm([
                [v[0], -v[1], 0],
                [v[1],  v[0], 0],
                [   0,     0, 1],
            ])
            quat = r.as_quat()
            # XYZW -> WXYZ order of elements
            quat = quat[[3,0,1,2]]
            
            detection_score = float(sample_detection_scores[i])

            
            box3d = Box3D(
                sample_token=sample_token,
                translation=list(translation),
                size=list(size),
                rotation=list(quat),
                name=class_name,
                score=detection_score
            )
            pred_box3ds.append(box3d)

def clean_up(train_data_folder,validation_data_folder):
    shutil.rmtree(train_data_folder)
    shutil.rmtree(validation_data_folder)