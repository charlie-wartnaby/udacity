def LeNet(x, num_classes, num_colours, layer1_depth):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu_w = 0
    sigma_w = 0.1 # Initial weights clustered around zero reasonably tightly
    mu_b = 0
    sigma_b = 0.02 # I have a little bit of noise in the initial biases too, experimentally
    
    # TODO: Layer 1: Convolutional. Input = 32x32x[colour chans]. Output = 28x28xlayer1_depth.
    # CW: if the output is 28x28, implies we have a 5x5 filter patch
    layer1_weights = tf.Variable(tf.truncated_normal(shape=(5,5,num_colours,layer1_depth), mean=mu_w, stddev=sigma_w), name='layer1_weights')
    layer1_biases = tf.Variable(tf.truncated_normal([layer1_depth], mu_b, sigma_b), name='layer1_biases')
    layer1_strides = [1, 1, 1, 1] # i.e. no striding
    layer1_padding = 'VALID' # hence we went down from 32 to 28
    layer1_conv_pre_bias = tf.nn.conv2d(x, layer1_weights, strides=layer1_strides,
                                        padding=layer1_padding, name='layer1_conv_pre_bias')
    layer1_conv_post_bias = tf.add(layer1_conv_pre_bias,layer1_biases, name='layer1_conv_post_bias')

    # TODO: Activation.
    # CW: no direction as to function so guessing
    layer1_activation = tf.nn.relu(layer1_conv_post_bias)

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x12.
    layer1_pool_ksize = [1,2,2,1] # batch, 2x2 height/width, pool x/y only
    layer1_pool_strides = [1, 2, 2, 1] # batch, 2x2 height/width stride, 1=depth stride
    layer1_pool_padding = 'VALID'
    layer1_pooled = tf.nn.max_pool(layer1_activation, layer1_pool_ksize, 
                                   layer1_pool_strides, layer1_pool_padding)

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    # CW: if the output is 10x10 from 14x14, implies we have a 5x5 filter patch
    layer2_weights = tf.Variable(tf.truncated_normal([5,5,layer1_depth,16], mu_w, sigma_w))
    layer2_biases = tf.Variable(tf.truncated_normal([16], mu_b, sigma_b))
    layer2_strides = [1, 1, 1, 1] # i.e. no striding
    layer2_padding = 'VALID' # hence we went down from 14 to 10
    layer2_conv_pre_bias = tf.nn.conv2d(layer1_pooled, layer2_weights, strides=layer2_strides,
                                        padding=layer2_padding)
    layer2_conv_post_bias = layer2_conv_pre_bias + layer2_biases
    
    # TODO: Activation.
    # CW: no direction as to function so guessing
    layer2_activation = tf.nn.relu(layer2_conv_post_bias)

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    layer2_pool_ksize = [1,2,2,1] # batch, 2x2 height/width, output depth pooling xy only
    layer2_pool_strides = [1, 2, 2, 1] # batch, 2x2 stride in 1=height/depth only
    layer2_pool_padding = 'VALID'
    layer2_pooled = tf.nn.max_pool(layer2_activation, layer2_pool_ksize, 
                                   layer2_pool_strides, layer2_pool_padding)

    # TODO: Flatten. Input = 5x5x16. Output = 400.
    layer2_flattened = tf.contrib.layers.flatten(layer2_pooled)
    
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    layer3_weights = tf.Variable(tf.truncated_normal([400,120], mu_w, sigma_w))
    layer3_biases =tf.Variable( tf.truncated_normal([120], mu_b, sigma_b))
    layer3_pre_bias = tf.matmul(layer2_flattened,layer3_weights)
    layer3_post_bias = layer3_pre_bias + layer3_biases
    
    # TODO: Activation.
    # CW: no direction as to function so guessing
    layer3_activation = tf.nn.relu(layer3_post_bias)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    layer4_weights = tf.Variable( tf.truncated_normal([120,84], mu_w, sigma_w))
    layer4_biases = tf.Variable(tf.truncated_normal([84], mu_b, sigma_b))
    layer4_pre_bias = tf.matmul(layer3_activation, layer4_weights)
    layer4_post_bias = layer4_pre_bias + layer4_biases
    
    # TODO: Activation.
    # CW: no direction as to function so guessing
    layer4_activation = tf.nn.relu(layer4_post_bias)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = num_classes.
    layer5_weights = tf.Variable(tf.truncated_normal([84,num_classes], mu_w, sigma_w))
    layer5_biases = tf.Variable(tf.truncated_normal([num_classes], mu_b, sigma_b))
    layer5_pre_bias = tf.matmul(layer4_activation, layer5_weights)
    layer5_post_bias = layer5_pre_bias + layer5_biases

    # I had softmax but solution used output directly
    logits = tf.nn.softmax(layer5_post_bias, name='logits')
    #logits = layer5_post_bias
    
    return logits

# This is required by both training and final testing cells
def evaluate(sess, accuracy_operation, X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    #sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

### Train your model here.
import numpy as np

def debug_to_gray(x):
    "For some experiments converted colour image to greyscale"
    gray_array = []
    for i in range(len(x)):
        rgb_image = x[i]
        gray_image = np.dot(rgb_image, [0.299,0.587,0.114]).reshape((32,32,1))
        x[i] = gray_image
        gray_array.append(gray_image)
    return gray_array
        
def debug_limit_classes(x_set, y_set, max_classes):
    "Restrict to a smaller number of classes to make training easier during debugging"
    limited_x = []
    limited_y = []
    for x,y in zip(x_set, y_set):
        if y < max_classes:
            limited_x.append(x)
            limited_y.append(y)
    return limited_x, limited_y


# Start by loading previously normalised data
import pickle

print("Retrieving normalised data from file...")
with open("normalised_data.p", mode='rb') as f:
    x_train_norm = pickle.load(f)
    y_train      = pickle.load(f)
    x_valid_norm = pickle.load(f)
    y_valid      = pickle.load(f)
    x_test_norm  = pickle.load(f)
    y_test       = pickle.load(f)
print("... data loaded.")

##############################################
#  Hyperparameters
num_colours = 3  # 1 greyscale, 3 colour
EPOCHS = 700
BATCH_SIZE = 256
# rate = 0.0003 # now experimenting with dynamic learning rate
layer1_depth=18 # LeNet was 6, experimenting
##############################################

if num_colours < 3:
    x_train_norm = debug_to_gray(x_train_norm)
    x_valid_norm = debug_to_gray(x_valid_norm)
    x_test_norm = debug_to_gray(x_test_norm)

#debug_limit_max_classes = 5
#x_train_norm, y_train = debug_limit_classes(x_train_norm, y_train, debug_limit_max_classes)
#x_valid_norm, y_valid = debug_limit_classes(x_valid_norm, y_valid, debug_limit_max_classes)
#x_test_norm, y_test = debug_limit_classes(x_test_norm, y_test, debug_limit_max_classes)

# shuffle as per LeNet exercise
from sklearn.utils import shuffle
x_train, y_train = shuffle(x_train_norm, y_train)

# Find range of class label numbers so we can map them to a one-hot vector;
# in the initial data analysis, we already checked that all of the labels
# occur in the training set
y_min = np.min(y_train)
y_max = np.max(y_train)
num_classes = y_max - y_min + 1
print("num_classes = " + str(num_classes))


import tensorflow as tf
import sys

#CW define placeholders for input training data
x = tf.placeholder(tf.float32, (None, 32, 32, num_colours), name='x')
y = tf.placeholder(tf.int32, (None), name='y')

# Now using dynamic learning rate
dynamic_rate = tf.placeholder(tf.float32, (None), name='dynamic_rate')

#CW map output class numbers to one-hot encoded vector
one_hot_y = tf.one_hot(y, num_classes, name='one_hot_y')

# Training pipeline from LeNet lab
logits = LeNet(x, num_classes, num_colours, layer1_depth)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits, name='cross_entropy')
loss_operation = tf.reduce_mean(cross_entropy, name='loss_operation')
optimizer = tf.train.AdamOptimizer(learning_rate = dynamic_rate, name='optimizer')
training_operation = optimizer.minimize(loss_operation, name='training_operation')

# evaluation from LeNet
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1), name='correct_prediction')
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')
saver = tf.train.Saver()


### Calculate and report the accuracy on the training and validation set.

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)
    
    print("Training with EPOCHS=%d BATCH_SIZE=%d rate=<dynamic> num_colours=%d layer1 depth=%d" % (EPOCHS, BATCH_SIZE, num_colours, layer1_depth))
    print()
    # tab-delimited file headings
    print("Epochs\tValidation Accuracy")

    for epoch_idx in range(EPOCHS):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            rate = 0.1 / (epoch_idx + 100) # start fast, slow down to get better ultimate convergence
            (to, lo, ohy) = sess.run((training_operation, logits, one_hot_y), feed_dict={x: batch_x, y: batch_y, dynamic_rate: rate})
            # logits and one-hot y were also extracted there for debugging
            
        validation_accuracy = evaluate(sess, accuracy_operation, x_valid_norm, y_valid)
        # Output results in tab-delimited format to make import into Excel easier
        report_str = "%d\t%f" % ((epoch_idx+1), validation_accuracy)
        # writing to stdout (so I could redirect to file for import into Excel when running on command line)
        # and stderr (so I could see progress at the same time)
        print(report_str)
        sys.stderr.write(report_str + "\n")
        
    # Encode hyperparameters in model filename so we can keep resulting model from different experiments
    # and then go back and use best one in final testing
    save_filename = 'traffic_sign_classifier_model-E%d-B%d-R%f-C%d-L1D%d' % (EPOCHS, BATCH_SIZE, rate, num_colours, layer1_depth)
    saver.save(sess, './' + save_filename)
    print("Model saved as " + save_filename)

