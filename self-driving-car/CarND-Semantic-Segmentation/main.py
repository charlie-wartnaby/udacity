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
from PIL import Image
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


def load_vgg(keep_prob):
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
    library_model = tf.keras.applications.VGG16(weights='imagenet') # But this gives tensorflow.python.keras.engine.functional.Functional object fetched from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    # Note: that gets cached in <user>\.keras\models
    print("VGG model as initially loaded:")
    library_model.summary() # works with Functional object but not AutoTrackable

    for layer in library_model.layers:
        layer.Trainable = False

    library_model.trainable = False 

    # Tried returning model without insertion of dropout layers, but still
    # not learning correctly so not that
    #return library_model

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
    drop_prob = 1.0 - keep_prob
    dropout1 = tf.keras.layers.Dropout(drop_prob, name="dropout1")
    dropout2 = tf.keras.layers.Dropout(drop_prob, name="dropout2")
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
    epochs = 2 if quick_run_test else 100 # Model pretty much converged after this time and no apparent overtraining
    batch_size = 1 if quick_run_test else 8 # 6 fitted my Quadro P3000 device without memory allocation warning
    keep_prob = 0.5  # Currently makes no difference if 0 or 1 so not working
    learning_rate = 0.005

    # Load pretrained VGG16 including dropout layers not included in standard Keras version
    model = load_vgg(keep_prob)

    # CW: add our own layers to do transpose convolution skip connections from encoder
    model = add_layers(model, num_classes) # get final layer out

    #opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Original
    #opt = tf.keras.optimizers.Adadelta(learning_rate=0.05) # About 80% accuracy but result images still look random
    #opt = tf.keras.optimizers.Adagrad(learning_rate=0.01)
    opt = tf.keras.optimizers.Ftrl(learning_rate=0.005) # Much better loss (~0.5) but output still looks random
    #opt = tf.keras.optimizers.Nadam() # Equally poor as most others
    #opt = tf.keras.optimizers.SGD() # Blew up
    #opt = tf.keras.optimizers.Adamax() # No better
    #opt = tf.keras.optimizers.RMSprop() # No better
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

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

    if False:
        # Debug: re-emitting training data as images to check still looks reasonable. Which it does, so
        # input processing doesn't seem to be the problem
        for i in range(num_images):
            x_debug_fn = helper.gen_batch_function(data_path, image_shape, num_classes, batch_size, quick_run_test)
            x_debug_data = next(x_debug_fn)
            image_array = x_debug_data[0][0]
            class_array = x_debug_data[1][0]
            road_image = Image.fromarray(image_array, mode='RGB')
            road_image.save(r'c:\temp\road_image' + str(i) + '.png')
            for j in range(2):
                class_array_class = class_array[:,:,j]
                class_array_scaled = np.uint8(class_array_class * 255) # PIL greyscale load gives odd results unless uint8 input
                class_image = Image.fromarray(class_array_scaled, mode='L')
                class_image.save(r'c:\temp\class_image_' + str(j) + '_' + str(i) + '.png')

    model.fit(x=helper.gen_batch_function(data_path, image_shape, num_classes, batch_size, quick_run_test),
              batch_size=batch_size,
              steps_per_epoch = num_images / batch_size,
              epochs=epochs)

    helper.save_inference_samples(runs_dir, data_dir, model, image_shape, quick_run_test)

    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
