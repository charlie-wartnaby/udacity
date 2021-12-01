import re
import random
import numpy as np
import os.path
import shutil
import zipfile
import time
import tensorflow as tf
from tensorflow import keras
import torch
from glob import glob
from PIL import Image
from urllib.request import urlretrieve
from tqdm import tqdm

import torch_vgg


def get_image_paths(data_folder, quick_run_test, include_gt):
    # CW: Kitti dataset. Each photo has two label images.
    #  data/data_road/image/training/image_2 = collection of .png colour photos of roads e.g. umm_000042.png
    #  data/data_road/image/training/gt_image_2 = .png of ground truth (gt) where magenta=our lane/road,
    #                              black=other road, red=something else
    #      e.g. umm_road_000042.png for ground truth whole road, umm_lane_000042.png = just our lane
    #  prefixes um_, umm_, uu_: http://www.cvlibs.net/datasets/kitti/eval_road.php
    #       uu - urban unmarked (98/100)
    #       um - urban marked (95/96)
    #       umm - urban multiple marked lanes (96/94)

    ip_folder = os.path.join(data_folder, 'image_2')     # input images
    gt_folder = os.path.join(data_folder, 'gt_image_2')  # ground truth images

    image_paths = glob(os.path.join(ip_folder, '*.png')) # so raw photos
    if quick_run_test:
        image_paths[10:] = []

    # Get corresponding ground truth image filenames (labels)
    gt_image_paths = []
    for path in image_paths:
        if include_gt:
            ip_filename = os.path.basename(path)
            gt_filename = re.sub(r'([a-z]+)_(.+)', r'\1_road_\2', ip_filename)
            gt_path = os.path.join(gt_folder, gt_filename)
            gt_image_paths.append(gt_path)
        else:
            gt_image_paths.append(None)

    return list(zip(image_paths, gt_image_paths))

def keras_normalise_image(image_array):
    # Never did input data normalisation in original project! This should do roughly, untested:
    converted_array = image_array.astype(np.float32)
    converted_array /= 255.0
    converted_array -= 0.5
    return converted_array

def gen_batch_function(image_paths, image_shape, num_classes, batch_size):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    num_images = len(image_paths)

    while True: # Keep producing batches of data no matter how many epochs run
        random.shuffle(image_paths)
        for batch_i in range(0, num_images, batch_size): # divide full (shuffled) set into batches
            ip_images = []
            gt_images = []
            for ip_image_file, gt_image_file in image_paths[batch_i:batch_i+batch_size]:
                ip_image, gt_image = form_image_arrays(ip_image_file, gt_image_file, image_shape)

                ip_image = keras_normalise_image(ip_image)

                ip_images.append(ip_image) 
                gt_images.append(gt_image)  # so now have 4D for ground truth, i.e. [image, height, width, classes] (or w,h, not sure)

            yield (np.array(ip_images), np.array(gt_images)) # return this batch


def form_image_arrays(image_file, gt_image_file, image_shape):
    """Load input and ground truth images from disk and convert to usable Numpy arrays"""

    background_color = np.array([255, 0, 0]) # CW: red
                
    unscaled_image = Image.open(image_file)       # real photo
    image = np.array(unscaled_image.resize(image_shape))
    
    if gt_image_file:
        unscaled_gt_image = Image.open(gt_image_file)       # ground truth image
        gt_image = np.array(unscaled_gt_image.resize(image_shape))

        gt_bg = np.all(gt_image == background_color, axis=2) # CW: for each pixel, is it background (red)?
        
        # CW: for each pixel now in 2D array, adding a further array dimension spanning classes -- but
        #     only have one class, for road/not road, so has size 1 to start with:
        gt_bg = gt_bg.reshape(*gt_bg.shape, 1) # unpacks existing shape tuple as positional arguments -- adding a dimension of size 1?
        # ... so now have 3 dimensions for each image (height, width, classes)

        # ... then add element to each class dimension for opposite
        gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2) # inverted so 1=non-background
        # CW: so for each point, now have one-hot array like [0,1] (not background, is road) or
        #                                                    [1,0] (is background, not road)

        # Convert Booleans to float (actually Keras works OK without doing this anyway,
        # Torch loss function does not)
        gt_image = gt_image.astype(np.float32)
        
        # TODO -- so network will identify 'other' roads, (black in ground truth images),
        #         not just 'our' road (magenta in images) -- is that OK/intended?

    else:
        # For testing there is no ground truth image
        gt_image = None

    return image, gt_image


def gen_test_output_keras(model, input_image_paths, image_shape):
    """
    Generate test output using the test images
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in input_image_paths:
        images = []
        unscaled_image = Image.open(image_file)
        scaled_image = unscaled_image.resize(image_shape)
        normalised_image = keras_normalise_image(scaled_image)
        images.append(np.array(normalised_image))

        softmax_predictions = model.predict(np.array(images))

        yield softmax_predictions

def gen_test_output_torch(model, input_image_paths, image_shape):
    """
    Generate test output using the test images
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
        # Create dataset object which gets array data from disk files
    dataset = torch_vgg.TorchDataset(input_image_paths, image_shape, False)

    # And DataLoader for preprocessing
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval() # Sets mode for dropout layers etc
    use_gpu = torch.cuda.is_available()

    for iter, batch in enumerate(dataloader):
        if use_gpu:
            inputs = torch.autograd.Variable(batch[0].cuda()) # Source example indexed batch as dict though, 'X' & 'Y'
        else:
            inputs = torch.autograd.Variable(batch[0])

        softmax_predictions = model(inputs)

        yield softmax_predictions

def save_inference_samples(framework, runs_dir, data_dir, model, image_shape, quick_run_test):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    input_folder = os.path.join(data_dir, 'data_road/testing')
    image_paths = get_image_paths(input_folder, quick_run_test, False)

    if framework == "keras":
        inference_outputs = gen_test_output_keras(model,
                                                     image_paths, image_shape)
    else:
        inference_outputs = gen_test_output_torch(model,
                                                  image_paths, image_shape)
        pass

    count = 0
    for ip_image_file, softmax_predictions in zip(image_paths, inference_outputs):
        unscaled_image = Image.open(ip_image_file)
        softmax_prediction = softmax_predictions[0]
        predicted_class0 = softmax_prediction[:,:,0]
        segmentation_flag = (predicted_class0 < 0.5)
        segmentation_array = np.reshape(segmentation_flag, (image_shape[0], image_shape[1], 1))
        mask = np.dot(segmentation_array, np.array([[0, 255, 0, 127]], dtype=np.uint8))
        mask = Image.fromarray(mask, mode="RGBA")
        rescaled_mask = mask.resize(unscaled_image.size)
        street_im = unscaled_image
        street_im.paste(rescaled_mask, box=None, mask=rescaled_mask)
        filename = os.path.basename(ip_image_file)
        street_im.save(os.path.join(output_dir, filename))
        count += 1
        if quick_run_test and count >= 5:
            break
