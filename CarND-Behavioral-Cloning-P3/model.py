"""
Self-driving car course: behavioural cloning project, model training script

Author:  Charlie Wartnaby, Applus IDIADA
Email:   charlie.wartnaby@idiada.com
"""

################################################################################
# Load data from disk
################################################################################

import csv
import cv2
import os.path

# The raw data recordings are stored in several separate directories to
# allow experimentation with just a subset, and to allow more weight to
# be given to different types of data. Here the relative paths of the
# different subsets are specified together with a decimation factor,
# e.g. if decimation=10, then only 1 in 10 images will be used. That
# allowed quicker development of this program, exploiting the fact that
# successive images and steering values tend to be very similar anyway.
# Somewhat higher weight (smaller decimation factor) can be applied to
# the perhaps more important 'rare' recovery scenarios to try and ensure
# that the model uses those strongly to gain robustness to being off-track.
dirs_decimations = [(r"recordings\left_circuit_centred",  1), # normal circuit
                    (r"recordings\right_circuit_centred", 1), # reverse circuit
                    (r"recordings\recoveries",            1 )] # off-course recoveries

# In my own copy of the logfiles I have inserted headings, so we could use
# csv.DictReader to access the columns by name. But for future compatibility
# with heading-less logfiles, using hard-coded column indices here:
col_centre_path = 0
col_left_path   = 1
col_right_path  = 2
col_steering    = 3
col_throttle    = 4
col_brake       = 5
col_speed       = 6

centre_images = []    # retained raw images
steering_values = []  # corresponding steering values to learn from

# Whether to add a left-right flipped version of the image and negated
# steering value for each original image added, to give the model more
# to train on
add_flips = True

if add_flips:
    print("Adding horizontally flipped version of each image with negated steering value too")

for (relative_folder, decimation_factor) in dirs_decimations:

    # Use relative paths so this can be run elsewhere, e.g. a GPU machine instance
    csv_file_rel_path = os.path.join(relative_folder, "driving_log.csv")
    image_dir_rel_path = os.path.join(relative_folder, "IMG")

    print("Loading index data from %s retaining 1 in %d images" % (csv_file_rel_path, decimation_factor))
    csv_index_file_pointer = open(csv_file_rel_path)
    reader = csv.reader(csv_index_file_pointer)
    decimator_value = 0
    lines = []            # useful data lines from .csv index file

    for line in reader:
        if "jpg" in line[col_centre_path]: # skip headings, if any
            if decimator_value == 0:
                # keep this line if rolling decimator value is zero
                lines.append(line)
            # Roll decimator value; if factor is 1 will keep every image
            decimator_value += 1
            if (decimator_value >= decimation_factor):
                decimator_value = 0
    print("... %d lines retained from index file" % len(lines))
    for line in lines:
        steering_value = float(line[col_steering])
        centre_path_original = line[col_centre_path]
        (original_dir, centre_filename) = os.path.split(centre_path_original)
        centre_path_relative = os.path.join(image_dir_rel_path, centre_filename)
        image = cv2.imread(centre_path_relative)
        centre_images.append(image)
        steering_values.append(steering_value)
        if add_flips:
            flipped_image = cv2.flip(image, 0)
            negated_steering = -steering_value
            centre_images.append(flipped_image)
            steering_values.append(negated_steering)
    print("Now have %d images and corresponding steering values" % len(centre_images))


################################################################################
# Convert data to form for modelling
################################################################################

import numpy as np

X_train = np.array(centre_images)
y_train = np.array(steering_values)

# Although keras.fit() takes a shuffle parameter, that does not apply to validation
# data -- it is always the last portion of data held back for validation. So 
# if we have just added a chunk of test images, those won't help the model, they will
# merely become part of the validation set. So shuffle the entire dataset now
# to mix up the older and more recently acquired images. This was crucial!

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)


################################################################################
# Create and train model
################################################################################

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D

img_shape = np.shape(X_train[0])
img_height = img_shape[0]

model = Sequential()

# Crop just above the horizon (with a little leeway for when the car
# dips and bounces), and just above the car bonnet/hood. Judging by
# straight-ahead pictures, we could crop more aggressively at the bottom,
# but when going off-track there are big blocks of colour quite low down
# at the sides which look imporant. From analysing a sample of pictures,
# it looks about right to start at 60 pixels from the top edge of the picture
# and finish at 136 pixels from the top edge (as indicated in MS Paint).
# No left or right cropping is applied here as the information in the
# pictures looks important right to the edges, at least when going
# off-track when the big blocks of non-track should help the model
# detect that strong steering is required.
top_crop_from_top = 60
bottom_crop_from_top = 136
bottom_crop_from_bottom = img_height - bottom_crop_from_top
left_crop = 0
right_crop = 0
model.add(Cropping2D(cropping=((top_crop_from_top,bottom_crop_from_bottom), (left_crop,right_crop)), input_shape=img_shape))

# Normalise pixel RGB values to -0.5 to +0.5 range
model.add(Lambda(lambda x: x / 255.0 - 0.5))

# Final solution used modified NVIDIA architecture, but also experimented with LeNet
# with some success:
lenet = False
if lenet:
    # Try LeNet
    print("Using adapted LeNet architecture")
    #  Layer 1: Convolutional with 5x5 filter patch
    model.add(Convolution2D(6,5,5,activation="relu"))
    # Pooling across 2x2 patch
    model.add(MaxPooling2D())
    model.add(Dropout(0.3)) # CW added to avoid overfitting
    # Layer 2: Convolutional with 5x5 filter patch
    model.add(Convolution2D(6,5,5,activation="relu"))
    # Pooling across 2x2 patch
    model.add(MaxPooling2D())
    model.add(Dropout(0.3)) # CW added to avoid overfitting
    model.add(Flatten())
    # Layer 3: Fully Connected. Input = 400. Output = 120
    #model.add(Dense(400)) # Missing from video, did worse with it
    #model.add(Dropout(0.15))
    # Layer 4: Fully Connected. Input = 120. Output = 84
    model.add(Dense(120))
    model.add(Dropout(0.3)) # CW added to avoid overfitting
    # Layer 5: Fully Connected. Input = 84. Output = 10
    model.add(Dropout(0.3)) # CW added to avoid overfitting
    model.add(Dense(84))
    # Finally get to single steering value
    model.add(Dense(1))

else:
    # Nvidia example based on project tutorial video, but with some dropout added to
    # avoid overtraining
    print("Using adapted NVIDIA architecture")

    # Assuming Nvidia used higher resolution than 320x160, I experimented with cutting down the
    # size of the convolutional layers, as there is surely less input information from them to
    # learn from (this also sped up modelling of course!)
    resolution_divide_factor = 2

    # Also tried changing subsample here to 1,1 as another way of adjusting the model to work
    # with a coarser starting resolution (but settled on factor used throughout instead):
    model.add(Convolution2D(int(24/resolution_divide_factor),5,5,subsample=(2,2),activation='relu'))

    model.add(Convolution2D(int(36/resolution_divide_factor),5,5,subsample=(2,2),activation='relu'))
    model.add(Dropout(0.1)) # CW added to avoid overfitting
    model.add(Convolution2D(int(48/resolution_divide_factor),5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(int(64/resolution_divide_factor),3,3,activation='relu'))
    model.add(Dropout(0.1)) # CW added to avoid overfitting
    model.add(Convolution2D(int(64/resolution_divide_factor),3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.1)) # CW added to avoid overfitting
    model.add(Dense(50))
    model.add(Dropout(0.1)) # CW added to avoid overfitting
    model.add(Dense(10))
    model.add(Dense(1)) # CW to get final steering angle, not shown in tutorial I think

# Using Adam optimiser as that seems to work nicely, hence no explicit learning rate needed
model.compile(loss='mse', optimizer='adam')

# This was a sufficient number of epochs in the end, though validation accuracy was still
# improving slightly so could have made it higher -- and that suggested it wasn't overfitting,
# perhaps thanks to the Dropout layers added
model.fit(X_train, y_train, validation_split=0.25, shuffle=True, nb_epoch=6)

model.save("model.h5")

