############################################################################### 
#    Udacity self-driving car course : Semantic Segregation Project.
#
#   Author : Charlie Wartnaby, Applus IDIADA
#   Email  : charlie.wartnaby@idiada.com
#
############################################################################### 

import os.path
import tensorflow as tf
import helper
import numpy as np
import time
import datetime
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def load_vgg():
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)

https://github.com/Natsu6767/VGG16-Tensorflow/blob/master/vgg16.py shows KEEP_PROB dropout layer after fc1 and fc2

The Keras standard VGG16 doesn't have the dropout layers used by this project in the original
model provided. But see below for their insertion.

Output from model.summary() for tf.keras.applications.VGG16() to figure out new layer names:
 Layer (type)                 Output Shape              Param #     Old project name/comments
=================================================================--===========================================================
input_1 (InputLayer)         [(None, 224, 224, 3)]     0        'input_1:0' # Walkthrough: so we can pass through image
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         'layer3_out:0'  # Walkthrough: pool3 layer as shown in paper architecture
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         'layer4_out:0'  # Walkthrough: pool4 layer
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0          'layer7_out:0'  # Walkthrough: pool5 layer
flatten (Flatten)            (None, 25088)             0
fc1 (Dense)                  (None, 4096)              102764544  Note: no dropout layer in this library version
fc2 (Dense)                  (None, 4096)              16781312  
predictions (Dense)          (None, 1000)              4097000
 """
    vgg_tag = 'vgg16'    
 
    # Loading originally provided model for this project doesn't work; get AutoTrackable object without
    # the same interface as tensorflow.python.keras.engine.functional.Functional object got from library:
    #keras_loaded_model = tf.keras.models.load_model(vgg_path) # I'm sure model.summary() worked initially after this, but now get exception AutoTrackable object has no attribute summary
    #tf_loaded_model = tf.saved_model.load(vgg_path) # Seem to get identical AutoTrackable object with this call

    # TODO different sets of weights available for this...
    library_model = tf.keras.applications.VGG16() # But this gives tensorflow.python.keras.engine.functional.Functional object fetched from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    # Note: that gets cached in <user>\.keras\models
    print("VGG model as initially loaded:")
    library_model.summary() # works with Functional object but not AutoTrackable

    # Original tf v1
    # Following walkthrough tips
    #tf.compat.v1.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    #graph = tf.compat.v1.get_default_graph()

    # https://stackoverflow.com/questions/42475381/add-dropout-layers-between-pretrained-dense-layers-in-keras
    # Store the fully connected layers
    fc1         = library_model.get_layer("fc1")
    fc2         = library_model.get_layer("fc2")
    predictions = library_model.get_layer("predictions")
    # Create the dropout layers
    dropout1 = tf.keras.layers.Dropout(0.85, name="dropout1")
    dropout2 = tf.keras.layers.Dropout(0.85, name="dropout2")
    # Reconnect the layers
    x = dropout1(fc1.output)
    x = fc2(x)
    x = dropout2(x)
    predictors = predictions(x)
    # Create a new model
    mod_model = tf.keras.Model(inputs=library_model.input, outputs=predictors)

    # Display its architecture
    print("VGG model after inserting dropout layers:")
    mod_model.summary()

    return mod_model

# Not yet updated for v2 changes:
#tests.test_load_vgg(load_vgg, tf)


def add_layers(model, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # DONE: Implement function

    # See also lesson "FCN-8 Decoder" for structure, and Long_Shelhamer paper

    # Walkthrough video started with 1x1 convolution like this, but notes explained
    # that was already done for us (loaded model is not ordinary VGG but already
    # adapted for FCN). In fact the VGG network provided looks very much like
    # the one generated by the Single-Shot Detector caffe code, so I guess they
    # share some common heritage.
    #conv_1x1 = tf.layers.conv2d(vgg_layer7_out, # at/near end of VGG
    #                            num_classes, # just road/nonroad for us
    #                          1, # as 1x1 conv
    #                          padding='same',
    #                          kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3))

    # Using Tensorboard to visualise the structure of the VGG model provided, and
    # tf.trainable_variables() to list the dimensions and sizes of the weights and biases
    # for each layer, I arrive at this summary of what shape the output of each layer
    # is (knowing that we started with a 160 height x 576 width x 3 colour channel image).
    # All of the convolution layers have SAME padding and [1,1,1,1] strides so they
    # don't reduce the x-y pixel size. All the pooling layers have [1,2,2,1] strides so
    # they halve the pixel size. I'm ignoring the first dimension (across images), as
    # everything works on one image at a time.
    #
    # Layer name  Details                     Output dimensions
    # <input>     raw image                   160x576x3
    # conv1_1     conv2d 3x3x3x64, Relu       160x576x64
    # conv1_2     conv2d 3x3x64x64, Relu      160x576x64
    # pool1       pool [1,2,2,1]              80x288x64
    # conv2_1     conv2d 3x3x64x128, Relu     80x288x128
    # conv2_2     conv2d 3x3x128x128, Relu    80x288x128
    # pool2       pool [1,2,2,1]              40x144x128
    # conv3_1     conv2d 3x3x128x256, Relu    40x144x256
    # conv3_2     conv2d 3x3x256x256, Relu    40x144x256
    # conv3_3     conv2d 3x3x256x256, Relu    40x144x256
    # pool3       pool [1,2,2,1]              20x72x256     --> layer3_out
    # conv4_1     conv2d 3x3x256x512, Relu    20x72x512
    # conv4_2     conv2d 3x3x512x512, Relu    20x72x512
    # conv4_3     conv2d 3x3x512x512, Relu    20x72x512
    # pool4       pool [1,2,2,1]              10x36x512     --> layer4_out
    # conv5_1     conv2d 3x3x512x512, Relu    10x36x512
    # conv5_2     conv2d 3x3x512x512, Relu    10x36x512
    # conv5_3     conv2d 3x3x512x512, Relu    10x36x512
    # pool5       pool [1,2,2,1]              5x18x512
    # fc6         conv2d 7x7x512x4096, Relu   5x18x4096
    # dropout     dropout(keep_prob)          5x18x4096
    # fc7         conv2d 1x1x4096x4096, Relu  5x18x4096
    # dropout_1   dropout(keep_prob)          5x18x4096     --> layer7_out
    # layer8      conv2d_t                    10x36

    layer3_out  = model.get_layer('block3_pool').output
    layer4_out  = model.get_layer('block4_pool').output
    layer7_out  = model.get_layer('block5_pool').output

    # Upsample by 2. We need to work our way down from a kernel depth of 4096
    # to just our number of classes (i.e. 2). Should we do this all in one go?
    # Or keep more depth in as we work upwards? For now doing it all in one hit.
    layer8 = tf.keras.layers.Conv2DTranspose(num_classes, #filters, 
                                             4, # kernel size taken from classroom example, might experiment
                                             strides=2, # stride causes upsampling
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer8')
                   # tf.compat.v1.layers.conv2d_transpose(vgg_layer7_out, #inputs
                   #                     num_classes, # filters so going down from 4096 to 2, is this a good idea yet?!
                   #                     4, # kernel size taken from classroom example, might experiment
                   #                     2, # stride causes upsampling
                   #                     padding='same',
                   #                     kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                   #                     name='layer8')

    

    # Now we're at 10x36x2 so we have same pixel resolution as layer4_out. Can't directly add
    # in layer4_out because it has filter depth of 512. (Though we could have had our transpose
    # convolution only downsample to 512 for compatibility... might try that later)

    # Squash layer4 output with 1x1 convolution so that it has compatible filter depth (i.e. num_classes)
    layer4_squashed = tf.keras.layers.Conv2D(num_classes, # new number of filters,
                                             1,    # 1x1 convolution so kernel size 1
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer4_squashed')
    #layer4_squashed = tf.compat.v1.layers.conv2d(vgg_layer4_out,
    #                                   num_classes, # new number of filters
    #                                   1,    # 1x1 convolution so kernel size 1
    #                                   padding='same',
    #                                   kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
    #                                   name='layer4_squashed')

    # upsample by 2
    layer9 = tf.keras.layers.Conv2DTranspose(num_classes, # filters
                                             4, # kernel size taken from classroom example
                                             strides=(2,2), # stride causes upsampling
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer9')
    #layer9 = tf.compat.v1.layers.conv2d_transpose(layer8_plus_layer4,
    #                                    num_classes,
    #                                    4, # kernel size taken from classroom example, might experiment
    #                                    2, # stride causes upsampling
    #                                    padding='same',
    #                                    kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
    #                                    name='layer9')

    # Now we're at 20x72x2 so same pixel resolution as layer3_out, but need to squash that from
    # 256 filters to 2 (num_classes) before we can add it in as skip connection
    layer3_squashed = tf.keras.layers.Conv2D(num_classes, # new number of filters
                                             1,    # 1x1 convolution so kernel size 1
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer3_squashed')
   #layer3_squashed = tf.compat.v1.layers.conv2d(vgg_layer3_out,
    #                                   num_classes, # new number of filters
    #                                   1,    # 1x1 convolution so kernel size 1
    #                                   padding='same',
    #                                   kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
    #                                   name='layer3_squashed')


    # upsample by 8 to get back to original image size
    layer10 = tf.keras.layers.Conv2DTranspose(num_classes,
                                              32, # Finding quite large kernel works nicely
                                              strides=(8,8), # stride causes upsampling
                                              padding='same',
                                              kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                              name='layer10')
    #layer10 = tf.compat.v1.layers.conv2d_transpose(layer9_plus_layer3,
    #                                    num_classes,
    #                                    32, # Finding quite large kernel works nicely
    #                                    8, # stride causes upsampling
    #                                    padding='same',
    #                                    kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
    #                                    name='layer10')

    # so now we should be at 160x576x2, same as original image size, 2 classes

    # Connect the layers
    x1 = layer8(layer7_out)
    x2 = layer4_squashed(layer4_out)

    # now we can add skip layer of this dimension taken from corresponding encoder layer
    layer8_plus_layer4 = tf.keras.layers.add([x1, x2], name='layer8_plus_layer4')
    #layer8_plus_layer4 = tf.add(layer8, layer4_squashed, name='layer8_plus_layer4')

    x1 = layer9(layer8_plus_layer4)
    x2 = layer3_squashed(layer3_out)

    # now we can add skip layer of this dimension taken from corresponding encoder layer
    layer9_plus_layer3 = tf.keras.layers.add([x1, x2], name='layer9_plus_layer3')
    #layer9_plus_layer3 = tf.add(layer9, layer3_squashed, name='layer9_plus_layer3')

    predictors = layer10(layer9_plus_layer3)  # layer 10 should be same size as image

    # Create a new model
    mod_model = tf.keras.Model(inputs=model.input, outputs=predictors)
    print("Model after adding decoder layers:")
    mod_model.summary()

    return mod_model

# Not updated for v2 yet:
#tests.test_layers(add_layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # DONE: Implement function

    # Walkthrough video help from 17:30

    # See also lesson FCN-8 - Classification & Loss

    # have to reshape tensor to 2D to get logits.
    # Naming tensors to make debug easier if necessary
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name='logits')
    
    # Reshape labels before feeding to TensorFlow session

    # Similar code to traffic sign classifier project now:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf.stop_gradient(correct_label), name='cross_entropy')
    cross_entropy_loss = tf.reduce_mean(input_tensor=cross_entropy, name='cross_entropy_loss')
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, name='optimizer')
    train_op = optimizer.minimize(cross_entropy_loss, name='train_op')

    return (logits, train_op, cross_entropy_loss)
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # DONE: Implement function

    keep_prob_value = 0.5 # After experimentation this high rate eventually does better
    learning_rate_value = 0.001 # From experiments

    # Walkthrough video help from 19:30
    start_time = time.time()
    for epoch in range(epochs):
        total_loss = 0
        batches_run = 0
        for image, label in get_batches_fn(batch_size):
            # Labels are 4D [N image, height, width, classes]; we just want
            # to span pixels overall and classes for comparison with the network output

            # A note to self on sizing: Tensorflow does seem to handle the last batch being
            # smaller in size than the others, and so we can feed less data to a placeholder than
            # its allocated size, and get a smaller array out. E.g.
            # image.shape= (12, 160, 576, 3)   for a batch of 12 images x height x width x colour channels
            # but with 289 samples, the last one is:
            # image.shape= (1, 160, 576, 3)
            # and at output, we get corresponding logits_out.shape= (1105920, 2) and logits_out.shape= (92160, 2)
            # respectively, where 12*160*576=1105920 and 1*160*576=92160.

            # Construct feed dictionary
            feed_dict = {'image_input:0'   : image,
                         'correct_label:0' : label,
                         'keep_prob:0'     : [keep_prob_value],
                         'learning_rate:0' : learning_rate_value,
                         };

            # Then actually run optimizer and get loss (OK to do in one step? Seems to work OK.)
            train_out, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            batches_run += 1
            total_loss += loss
            print('.', end='', flush=True) # Show progress through batches

        elapsed_time = str(datetime.timedelta(seconds=(time.time() - start_time)))
        print("")
        print("Epoch:", epoch, "Loss/batch:", total_loss / batches_run, "time so far:", elapsed_time)

    print("")

tests.test_train_nn(train_nn)

def fake_generator():
    inputs = [0 * 10]
    targets = [1 * 10]
    X = np.array(inputs, dtype='float32')
    y = np.array(targets, dtype='float32')
    while True:
        yield (X, y)
 
def run():
    num_classes = 2 #  CW: just road or 'other'

    # CW: originals are 1242x375 so we are using shrunk and somewhat squashed versions
    # (more extreme letterbox aspect ratio than originals). Shrinking will reduce training
    #  workload.
    image_shape = (224,224) # Updated for keras library VGG16 (576, 160)  # width, height to fit Pillow.Image (prev scipy.image usage transposed)

    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    quick_run_test = False # For debug

    # Walkthrough: maybe ~6 epochs to start with. Batches not too big because large amount of information.
    epochs = 2 if quick_run_test else 50 # Model pretty much converged after this time and no apparent overtraining
    batch_size = 1 if quick_run_test else 8 # 6 fitted my Quadro P3000 device without memory allocation warning
    # Other hyperparameters in train_nn(); would have put them here but went with template calling structure

    # Load pretrained VGG16 including dropout layers not included in standard Keras version
    model = load_vgg()

    # Fetch tensors for input and output by name https://stackoverflow.com/questions/41711190/keras-how-to-get-the-output-of-each-layer
    image_input = model.get_layer('input_1').input
    keep_prob   = model.get_layer('dropout1').output

    # CW: add our own layers to do transpose convolution skip connections from encoder
    model = add_layers(model, num_classes) # get final layer out

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Create function to get batches
    #get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape, num_classes, quick_run_test)

    # OPTIONAL: Augment Images for better results
    #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

    # Walkthrough: correct labels will be 4D (batch, height, width, num classes)
    # CW: see my comments in get_batches_fn() to remind self of why... final (num classes) axis is one-hot
    #     with [0]=1 for background and [1]=1 for (any) road

    data_path = os.path.join(data_dir, 'data_road/training')
    num_images, image_paths = helper.get_image_paths(data_path, quick_run_test)
    model.fit(x=helper.gen_batch_function(data_path, image_shape, num_classes, batch_size, quick_run_test),
              batch_size=batch_size,
              steps_per_epoch = num_images / batch_size)

    helper.save_inference_samples(runs_dir, data_dir, model, image_shape, quick_run_test)

    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
