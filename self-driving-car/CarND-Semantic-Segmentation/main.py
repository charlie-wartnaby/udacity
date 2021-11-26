#   Email  : charlie.wartnaby@idiada.com
#
############################################################################### 

import os.path
import tensorflow as tf
import torch
import numpy as np
import time
import datetime
import warnings
from distutils.version import LooseVersion
from PIL import Image

import helper
import project_tests as tests
import torch_vgg

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


def keras_load_vgg():
    """
    Load Pretrained VGG Model

https://github.com/Natsu6767/VGG16-Tensorflow/blob/master/vgg16.py shows KEEP_PROB dropout layer after fc1 and fc2

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
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0                        # Walkthrough: pool5 layer
flatten (Flatten)            (None, 25088)             0     # Amorphous classifier from now on, no longer image shaped like course original
fc1 (Dense)                  (None, 4096)              102764544  Note: no dropout layers in this library version either
fc2 (Dense)                  (None, 4096)              16781312  
predictions (Dense)          (None, 1000)              4097000
 """
 
    # Loading originally provided model for this project doesn't work; get AutoTrackable object without
    # the same interface as tensorflow.python.keras.engine.functional.Functional object got from library:
    #keras_loaded_model = tf.keras.models.load_model(vgg_path) # I'm sure model.summary() worked initially after this, but now get exception AutoTrackable object has no attribute summary
    #tf_loaded_model = tf.saved_model.load(vgg_path) # Seem to get identical AutoTrackable object with this call

    # This gives tensorflow.python.keras.engine.functional.Functional object fetched from
    #  https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    library_model = tf.keras.applications.VGG16(weights='imagenet')
    # Note: that gets cached in <user>\.keras\models
    print("VGG model as initially loaded:")
    library_model.summary() # works with Functional object but not AutoTrackable

    # To avoid retraining pre-trained layers, make them untrainable. But doing this
    # for individual layers like this doesn't seem to work:
    #for layer in library_model.layers:
    #    layer.Trainable = False
    # Whereas making whole model (as it is first loaded) untrainable does seem to work,
    # with the custom added layers still being trainable. Odd.
    library_model.trainable = False 

    # Remove classifier layers which we'll replace with (small) image-shaped layers
    # like version of VGG16 originally used for this course
    block5_pool         = library_model.get_layer("block5_pool")
    # Create a new model skipping the classifier part of the library original
    mod_model = tf.keras.Model(inputs=library_model.input, outputs=block5_pool.output)

    # Display its architecture
    print("VGG model after removing classifier:")
    mod_model.summary()

    return mod_model

# Not yet updated for v2 changes:
#tests.test_load_vgg(load_vgg, tf)

def keras_add_layers(model, num_classes, keep_prob):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
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

    # Using Tensorboard to visualise the structure of the Udacity VGG model provided, and
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

    # Problem here: TF2 library model doesn't have image-shaped layers 6 & 7 like
    # model provided originally with TF1, but instead is flattened amporphous classifier.
    # So we're working with more 'raw' layer as input. TODO should add back
    # two conv2d layers before this to be like the original
    drop_prob = 1.0 - keep_prob

    layer5  = model.get_layer('block5_pool')

    layer6_conv = tf.keras.layers.Conv2D(4096,
                                         7,          # 7x7 patch from original Udacity model
                                         strides=(1,1),
                                         padding='same',
                                         kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)), # guess same as others
                                         name='layer6_conv')

    layer6_dropout = tf.keras.layers.Dropout(drop_prob, name="layer6_dropout")

    layer7_conv = tf.keras.layers.Conv2D(4096,
                                         1,         # 1x1 patch from original Udacity model
                                         strides=(1,1),
                                         padding='same',
                                         kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)), # guess
                                         name='layer7_conv')

    layer7_dropout = tf.keras.layers.Dropout(drop_prob, name="layer7_dropout")

    # Connect up the new layers
    x = layer6_conv(layer5.output)
    x = layer6_dropout(x)
    x = layer7_conv(x)
    layer7 = layer7_dropout(x)

    # Create a new model
    mod_model = tf.keras.Model(inputs=model.input, outputs=layer7)

    # We should now have the same structure as the original Udacity version of VGG16,
    # but still need to add the decoder and skip connections as before

    # Upsample by 2. We need to work our way down from a kernel depth of 4096
    # to just our number of classes (i.e. 2). Should we do this all in one go?
    # Or keep more depth in as we work upwards? For now doing it all in one hit.
    layer8 = tf.keras.layers.Conv2DTranspose(num_classes, #filters, 
                                             4, # kernel size taken from classroom example, might experiment
                                             strides=2, # stride causes upsampling
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer8')

    # Now we're at 10x36x2 so we have same pixel resolution as layer4_out. Can't directly add
    # in layer4_out because it has filter depth of 512. (Though we could have had our transpose
    # convolution only downsample to 512 for compatibility... might try that later)

    # Squash layer4 output with 1x1 convolution so that it has compatible filter depth (i.e. num_classes)
    layer4_squashed = tf.keras.layers.Conv2D(num_classes, # new number of filters,
                                             1,    # 1x1 convolution so kernel size 1
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer4_squashed')
    # upsample by 2
    layer9 = tf.keras.layers.Conv2DTranspose(num_classes, # filters
                                             4, # kernel size taken from classroom example
                                             strides=(2,2), # stride causes upsampling
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer9')

    # Now we're at 20x72x2 so same pixel resolution as layer3_out, but need to squash that from
    # 256 filters to 2 (num_classes) before we can add it in as skip connection
    layer3_squashed = tf.keras.layers.Conv2D(num_classes, # new number of filters
                                             1,    # 1x1 convolution so kernel size 1
                                             padding='same',
                                             kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                             name='layer3_squashed')

    # upsample by 8 to get back to original image size
    layer10 = tf.keras.layers.Conv2DTranspose(num_classes,
                                              32, # Finding quite large kernel works nicely
                                              strides=(8,8), # stride causes upsampling
                                              padding='same',
                                              kernel_regularizer = tf.keras.regularizers.l2(0.5 * (1e-3)),
                                              name='layer10')

    # so now we should be at 160x576x2, same as original image size, 2 classes

    # Connect the layers
    x1 = layer8(layer7)
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
    image_shape = (224,224) # Updated for keras library VGG16
                            # width, height to fit Pillow.Image (prev scipy.image usage transposed)

    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    quick_run_test = True # For debug

    # Walkthrough: maybe ~6 epochs to start with. Batches not too big because large amount of information.
    epochs = 2 if quick_run_test else 50 # Model pretty much converged after this time and no apparent overtraining
    batch_size = 1 if quick_run_test else 8 # 6 fitted my Quadro P3000 device without memory allocation warning
    keep_prob = 0.65 # In original project used high dropout rate (0.5), eventually better, but now struggling to converge unless higher 
    learning_rate = 0.001
    num_classes = 2 # road or not road
    framework = "torch"

    # Load pretrained VGG16
    if (framework == 'keras'):
        model = keras_load_vgg()
        # CW: add our own layers to do transpose convolution skip connections from encoder
        model = keras_add_layers(model, num_classes, keep_prob) # get final layer out
    else:
        model = torch_vgg.VggFcn(keep_prob=keep_prob, num_classes=num_classes)

    # Prepare model to run
    if (framework == 'keras'):
        #opt = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Original, converges slowly now needing small learning rate e.g. 0.0001 and 100+ epochs (250 was good but prob unnecessary, ~0.94 training accuracy though 1.5 loss)
        opt = tf.keras.optimizers.Ftrl(learning_rate=learning_rate) # better when didn't have VGG 7x7 classifier layers; loss ~1 (keep_prob=0.7) ~0.56 (kp=0.9) accuracy ~0.93 recently
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    else:
        pass

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


    if (framework == 'keras'):
        model.fit(x=helper.gen_batch_function(data_path, image_shape, num_classes, batch_size, quick_run_test),
                batch_size=batch_size,
                steps_per_epoch = num_images / batch_size,
                epochs=epochs)
    else:
        # Need Torch DataSet/TensorDaatSet to populate Torch tensors (can I reuse existing batch function
        # to load dynamically into Torch tensors?)
        # And DataLoader for preprocessing? 
        # https://discuss.pytorch.org/t/what-do-tensordataset-and-dataloader-do/107017
        # Then Torch style training
        pass

    helper.save_inference_samples(framework, runs_dir, data_dir, model, image_shape, quick_run_test)

    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
