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

    image_paths = glob(os.path.join(data_folder, 'image_2', '*.png')) # so raw photos

    num_images = 10 if quick_run_test else len(image_paths)

    return num_images, image_paths

def gen_batch_function(data_folder, image_shape, num_classes, batch_size, quick_run_test):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    num_images, image_paths = get_image_paths(data_folder, quick_run_test)

    # CW: Make dictionary to look up road (not lane) ground truth image for each photo
    #     e.g. umm_000042.png -> umm_road_000042.png
    label_paths = {
        re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
        for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

    background_color = np.array([255, 0, 0]) # CW: red

    random.shuffle(image_paths)
    for batch_i in range(0, num_images, batch_size): # divide full (shuffled) set into batches
        images = []
        gt_images = []
        for image_file in image_paths[batch_i:batch_i+batch_size]:
            gt_image_file = label_paths[os.path.basename(image_file)]

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

            # TODO -- so network will identify 'other' roads, (black in ground truth images),
            #         not just 'our' road (magenta in images) -- is that OK/intended?

            images.append(image)
            gt_images.append(gt_image)  # so now have 4D for ground truth, i.e. [image, height, width, classes] (or w,h, not sure)

        yield (np.array(images), np.array(gt_images)) # return this batch


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
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
        unscaled_image = Image.open(image_file)
        image = np.array(unscaled_image.resize(image_shape))

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[1], image_shape[0])
        segmentation = (im_softmax > 0.5).reshape(image_shape[1], image_shape[0], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]], dtype=np.uint8))
        mask = Image.fromarray(mask, mode="RGBA")
        street_im = Image.fromarray(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), street_im


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, quick_run_test):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    count = 0
    for name, image in image_outputs:
        image.save(os.path.join(output_dir, name))
        count += 1
        if quick_run_test and count >= 5:
            break
