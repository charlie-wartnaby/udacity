### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

# CW to start with I want to keep the images in colour, because I feel there is useful information
# encoded in that colour, e.g. the red border round speed signs.
# Looking at the example images though, the brightness varies a lot, so I will try and normalise
# for brightness, as well as getting the data into a near -1 to 1 range

import numpy as np

def normalise_one_image(colour_image_array):
    # Quick hack to do simpler normalisation for comparison:
    # new_image = colour_image_array / 128.0
    # new_image -= 1.0
    # return new_image

    (y_size, x_size, num_colours) = np.shape(colour_image_array)
    new_image = np.zeros(np.shape(colour_image_array))
    min_value = np.min(colour_image_array)
    max_value = np.max(colour_image_array)
    # Normalise so that darkest value is -1, brightest is +1, irrespective of colour
    observed_value_range = float(max_value - min_value)
    desired_min_value = -1.0
    desired_max_value = +1.0
    desired_range = desired_max_value - desired_min_value
    for y_coord in range(y_size):
        for x_coord in range(x_size):
            scale_factor = desired_range / observed_value_range
            for colour_idx in range(num_colours):
                # shift point colour values so that the lowest RGB value of any pixel in any
                # colour channel is zero
                pixel_colour_value = float(colour_image_array[y_coord][x_coord][colour_idx])
                pixel_colour_value -= min_value
                # scale so the range of intensities across the whole image matches our desired range
                pixel_colour_value *= scale_factor
                # offset so we get values in the desired range
                pixel_colour_value += desired_min_value
                new_image[y_coord][x_coord][colour_idx] = pixel_colour_value
        
    return new_image

def normalise_image_set(image_set_array):
    new_image_set = []
    debug_set_size_limit = 100000000
    count = 0
    for image in image_set_array:
        new_image = normalise_one_image(image)
        new_image_set.append(new_image)
        count += 1
        if count >= debug_set_size_limit:
            break
    return new_image_set

### Load the images and plot them here.
### Feel free to use as many code cells as needed.

import os
import matplotlib.image as mpimg

# Read in all images from here:
images_input_dir = "./signs_from_internet"

# Iterate over images, processing each one
image_filenames = os.listdir(images_input_dir)
internet_images = []
correct_internet_classes = []
internet_filenames = []

print ("Loading internet test images from disk...")
for image_filename in image_filenames:
    
    if image_filename.lower().endswith(".jpg"):
        # Read image from disk
        internet_filenames.append(image_filename)
        raw_path = os.path.join(images_input_dir, image_filename)
        raw_image = mpimg.imread(raw_path)
        internet_images.append(raw_image)
        
        # Each image is named with the correct class in decimal as a prefix,
        # e.g. "23-resized_slippery.jpg" is a slippery road caution which
        # has a correct category of 23 according to signnames.csv. We
        # record those class ID numbers so we can assess accuracy later.
        name_parts = image_filename.split('-')
        class_id = int(name_parts[0])
        correct_internet_classes.append(class_id)

print ("... %d images loaded." % len(internet_images))

# Load names of classes so they can be reported more intelligibly
import pandas
classes_descriptions_dataframe = pandas.read_csv("signnames.csv")
class_descriptions = []
for index, row in classes_descriptions_dataframe.iterrows():
    class_descriptions.append(row['SignName'])
print("%d class descriptions loaded" % len(class_descriptions))

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
print("Normalising images...")
processed_internet_images = normalise_image_set(internet_images)
print("... %d images normalised." % len(processed_internet_images))


import tensorflow as tf

# Started with evaluation section code fragment from LeNet lab

# Using best run-time parameters from previous experiments
num_colours = 3 # therefore no conversion to greyscale needed
layer1_depth = 24 # was not very sensitive to this so long as >>6 as per LeNet
num_classes = 43 # slight cheat to avoid restructuring and rerunning previous code above
BATCH_SIZE = 256 # should not have any effect on testing, we are not training any more

new_graph = tf.Graph()
with tf.Session(graph=new_graph) as sess:

    # https://stackoverflow.com/questions/33759623/tensorflow-how-to-save-restore-a-model
    # Restore the best model obtained during experiments
    # reload debug: traffic_sign_classifier_model-E1-B256-R0.001000-C3-L1D24
    # best: traffic_sign_classifier_model-E1000-B256-R0.000091-C3-L1D24
    model_name = 'traffic_sign_classifier_model-E700-B256-R0.000125-C3-L1D18'
    saver = tf.train.import_meta_graph(model_name + '.meta')
    saver.restore(sess, model_name)

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    logits = graph.get_tensor_by_name("logits:0")
    accuracy_operation = graph.get_tensor_by_name("accuracy_operation:0")
    print("Running model to classify images...")
    logits_out,accuracy_out = sess.run((logits,accuracy_operation), feed_dict={x: processed_internet_images, y: correct_internet_classes})
    print("... model run complete.")

	### Calculate the accuracy for these 5 new images. 
num_internet = len(processed_internet_images)
print("Final test accuracy on %d internet images = %.3f" % (num_internet, accuracy_out))
num_correct = int(accuracy_out * num_internet + 0.5) # float to int rounding
print("%d out of %d correctly classified" % (num_correct, num_internet))
#print("logits_out=" + str(logits_out))

### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.

import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

num_best_scores_per_sign = 5
print("Listing top %d ranked softmax probabilities per image (correct choice flagged with '*')" % num_best_scores_per_sign)
for image_idx in range(num_internet):
    logits_this_sign = logits_out[image_idx]
    logit_classidx = []
    for class_idx in range(len(logits_this_sign)):
        logit_classidx.append((logits_this_sign[class_idx],class_idx))
    logit_classidx.sort(reverse=True)
    print ("\nTop %d softmax probabilities for image filename %s:" %(num_best_scores_per_sign, internet_filenames[image_idx]))
    image = internet_images[image_idx]
    #plt.figure(figsize=(1,1))
    #plt.imshow(image)
    #plt.show() # Explicitly calling show() so it appears before next text line I want to output
    for idx in range(num_best_scores_per_sign):
        logit_value, class_idx = logit_classidx[idx]
        correct_flag = "*" if correct_internet_classes[image_idx] == class_idx else " "
        print("%s %d %.6f %s" % (correct_flag, idx+1, logit_value, class_descriptions[class_idx]))