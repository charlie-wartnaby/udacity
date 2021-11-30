import re
import random
import numpy as np
import os.path
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from PIL import Image
from urllib.request import urlretrieve
from tqdm import tqdm

def get_image_paths(data_folder, quick_run_test):
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

    num_images = 10 if quick_run_test else len(image_paths)

    # Get corresponding ground truth image filenames (labels)
    gt_image_paths = []
    for path in image_paths:
        ip_filename = os.path.basename(path)
        gt_filename = re.sub(r'([a-z]+)_(.+)', r'\1_road_\2', ip_filename)
        gt_path = os.path.join(gt_folder, gt_filename)
        gt_image_paths.append(gt_path)

    return num_images, list(zip(image_paths, gt_image_paths))


def gen_batch_function(data_folder, image_shape, num_classes, batch_size, quick_run_test):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    num_images, image_paths = get_image_paths(data_folder, quick_run_test)

    while True: # Keep producing batches of data no matter how many epochs run
        random.shuffle(image_paths)
        for batch_i in range(0, num_images, batch_size): # divide full (shuffled) set into batches
            ip_images = []
            gt_images = []
            for ip_image_file, gt_image_file in image_paths[batch_i:batch_i+batch_size]:
                ip_image, gt_image = form_image_arrays(ip_image_file, gt_image_file, image_shape)

                # TODO never did normalisation here! This should do roughly, untested:
                ip_image = ip_image.astype(np.float32)
                ip_image /= 255
                ip_image -= 0.5

                ip_images.append(ip_image) 
                gt_images.append(gt_image)  # so now have 4D for ground truth, i.e. [image, height, width, classes] (or w,h, not sure)

            yield (np.array(ip_images), np.array(gt_images)) # return this batch


def form_image_arrays(image_file, gt_image_file, image_shape):
    """Load input and ground truth images from disk and convert to usable Numpy arrays"""

    background_color = np.array([255, 0, 0]) # CW: red
                
    unscaled_image = Image.open(image_file)       # real photo
    image = np.array(unscaled_image.resize(image_shape))
    
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

    # Convert Booleans to float (actually works OK without doing this anyway)
    #gt_image = gt_image.astype(np.float32)
    
    # TODO -- so network will identify 'other' roads, (black in ground truth images),
    #         not just 'our' road (magenta in images) -- is that OK/intended?

    return image, gt_image


def gen_test_output(framework, model, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        images = []
        unscaled_image = Image.open(image_file)
        scaled_image = unscaled_image.resize(image_shape)
        images.append(np.array(scaled_image))

        if (framework == 'keras'):
            softmax_predictions = model.predict(np.array(images))
        else:
            pass
        
        softmax_prediction = softmax_predictions[0]
        predicted_class0 = softmax_prediction[:,:,0]
        segmentation_flag = (predicted_class0 < 0.5)
        segmentation_array = np.reshape(segmentation_flag, (image_shape[0], image_shape[1], 1))
        mask = np.dot(segmentation_array, np.array([[0, 255, 0, 127]], dtype=np.uint8))
        mask = Image.fromarray(mask, mode="RGBA")
        rescaled_mask = mask.resize(unscaled_image.size)
        street_im = unscaled_image
        street_im.paste(rescaled_mask, box=None, mask=rescaled_mask)

        yield os.path.basename(image_file), street_im


def save_inference_samples(framework, runs_dir, data_dir, model, image_shape, quick_run_test):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(framework,
        model, os.path.join(data_dir, 'data_road/testing'), image_shape)
    count = 0
    for name, image in image_outputs:
        image.save(os.path.join(output_dir, name))
        count += 1
        if quick_run_test and count >= 5:
            break
