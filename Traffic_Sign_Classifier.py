import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import random
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os.path
import math
import cv2

LOGDIR = "/tmp/sign_classifier/"
LABELS= "./signnames.csv"
SPRITES = "./traffic-sign-data/"

# Load pickled data
def load_pickled_data():
    training_file = "./traffic-sign-data/train.p"
    validation_file= "./traffic-sign-data/valid.p"
    testing_file = "./traffic-sign-data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def data_summary(X_train, X_valid, X_test):
    class_file_df = pd.read_csv(LABELS, delimiter=',')

    # Number of training examples
    n_train = np.shape(X_train)[0]

    # Number of validation examples
    n_validation = np.shape(X_valid)[0]

    # Number of testing examples.
    n_test = np.shape(X_test)[0]

    # What's the shape of an traffic sign image?
    image_shape = [np.shape(X_train)[1],np.shape(X_train)[2],np.shape(X_train)[3]]

    # How many unique classes/labels there are in the dataset.
    n_classes = class_file_df.shape[0]

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    return n_train, n_validation, n_test, image_shape, n_classes, class_file_df

def view_sample_image():# Let's look at what one of these pictures looks like
    # %matplotlib inline
    index = random.randint(0, len(X_train))
    image = X_train[index].squeeze()
    print(y_train[index])
    print(class_file_df['SignName'][y_train[index]])
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_subplot(111)
    ax.grid(b=False)
    plt.imshow(image)

def visualize_data(class_file_df):
    #Now lets see how many of each type I have in my training data
    plt.style.use('ggplot')
    # get the counts of signs in the training dataset
    unique, counts = np.unique(y_train, return_counts=True)
    # since the signs are sorted 1-42 we can just add the count to the dataframe
    class_file_df['train_count'] = pd.Series(counts,index=class_file_df.index)
    # And sor the dataframe by the counts of the signs to make the plot look nice
    class_file_df = class_file_df.sort_values(by=['train_count'])

    # setup the plot
    fig = plt.figure(figsize=(8,9))
    ax = fig.add_subplot(111)
    signs = class_file_df['SignName']
    y_pos = np.arange(len(signs))
    count = class_file_df['train_count']

    ax.barh(y_pos, count, align='center',
            color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(signs)
    ax.set_xlabel('count')
    ax.set_title('Training signs count')
    plt.show()

def images_to_gray(images):
    output = []
    for image in images:
        output.append(image_to_gray(image))
    return output;

def image_to_gray(img):
    """Applies the YUV transform"""
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return image

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def normalize(image):
    """Normalizes images"""
    channels = cv2.split(image)
    channel_len = len(channels)
    num_image_elements = 32*32*channel_len
    chan = []
    image_chan_mean, image_chan_stddev = cv2.meanStdDev(image)
    for i in range(channel_len):
        chan.append((channels[i]- image_chan_mean[i])/max(image_chan_stddev[i], 1./math.sqrt(num_image_elements)))
    return cv2.merge(chan)

def nomalize_batch(images):
    output = []
    for image in images:
        output.append(normalize(image))
    return output

def random_image_distortion(image):
    if random.randint(0,10) > 5:
        distorted_image = rotateImage(image)
    if random.randint(0,10) > 5:
        distorted_image = flip(image)
    else:
        return image
    return distorted_image

def rotateImage(image):
    #https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point#9042907
    angle = random.randint(-25,25)
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
    return result

def flip(image):
    horizontal_img = cv2.flip( image, 0 )
    return horizontal_img

def conv_layer(x, size_in, size_out, stride=1, name="conv"):
    with tf.name_scope(name):
        mu = 0
        sigma = 0.1
        w = tf.Variable(tf.truncated_normal([5,5,size_in,size_out],mu,sigma), name="W")
        b = tf.Variable(tf.truncated_normal([size_out],mu,sigma), name="b")
        conv = tf.nn.conv2d(x, w, strides=[1,stride,stride,1], padding="SAME")
        activation = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", activation)
        return tf.nn.max_pool(conv, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def fc_layer(x, size_in, size_out, stride=1, name="fc" ):
    with tf.name_scope(name):
        mu = 0
        sigma = 0.1
        w = tf.Variable(tf.truncated_normal([size_in,size_out],mu,sigma))
        b = tf.Variable(tf.truncated_normal([size_out]))
        fc = tf.matmul(x, w) + b
        activation = tf.nn.relu(fc + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", activation)
        return activation

def logit(x, size_in, size_out, name="logit" ):
    with tf.name_scope(name):
        mu = 0
        sigma = 0.1
        w = tf.Variable(tf.truncated_normal([size_in,size_out],mu,sigma))
        b = tf.Variable(tf.truncated_normal([size_out]))
        logit = tf.matmul(x, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", logit)
        return logit  

def ModelArch(X_train, y_train, X_valid, y_valid, X_test, y_test,\
              learn_rate, n_classes, hparam="test_run", use_dropout=False, dropout_prob=0,\
              image_channels=3):
    EPOCHS = 30
    BATCH_SIZE = 128
    tf.reset_default_graph()
    sess = tf.Session()

    # setup the feautures, labels
    # x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
    x = tf.placeholder(tf.float32, shape=[None, 32, 32, image_channels], name="x")
    tf.summary.image('input', x, 1)
    y = tf.placeholder(tf.int32, (None), name="labels")
    one_hot_y = tf.one_hot(y,n_classes)

    # setup the nn with 2 convolutional layers and two fully connected layers
    # also give the option to enable dropouts after each of the fc layers
    size_in = {
        'conv_layer1': image_channels,
        'conv_layer2': 6,
        'fc_layer1': 1024,
        'fc_layer2': 120,
        'logit': 84
    }
    size_out = {
        'conv_layer1': 6,
        'conv_layer2': 16,
        'fc_layer1': 120,
        'fc_layer2': 84,
        'logit': n_classes
    }
    conv1 = conv_layer(x, size_in['conv_layer1'],size_out['conv_layer1'], name="conv1")
    conv2 = conv_layer(conv1, size_in['conv_layer2'],size_out['conv_layer2'], name="conv2")
    conv2 = flatten(conv2)
    fc1 = fc_layer(conv2, size_in['fc_layer1'],size_out['fc_layer1'], name="FC1")
    if use_dropout:
        fc1 = tf.nn.dropout(fc1, dropout_prob)
    fc2 = fc_layer(fc1, size_in['fc_layer2'],size_out['fc_layer2'],name="FC2")
    if use_dropout:
        fc2 = tf.nn.dropout(fc2, dropout_prob)
    logits = logit(fc2,size_in['logit'],size_out['logit'],name="output")

    
    with tf.name_scope("cross-entropy"):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=one_hot_y), name="cross-entropy")
        tf.summary.scalar("cross-entropy", cross_entropy)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
        accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy_operation)

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
        training_operation = optimizer.minimize(cross_entropy)
    
    summ = tf.summary.merge_all()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    print("Training...")
    for i in range(EPOCHS):
        num_examples = len(X_train)
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            batch_x = [np.reshape(me, [32,32,image_channels]) for me in batch_x]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
        
        num_valid_examples = len(X_valid)
        validation_accuracy = 0
        for offset in range(0, num_valid_examples, BATCH_SIZE):
            batch_x, batch_y = X_valid[offset:offset+BATCH_SIZE], y_valid[offset:offset+BATCH_SIZE]
            batch_x = [np.reshape(me, [32,32,image_channels]) for me in batch_x]
            [accuracy,s] = sess.run([accuracy_operation,summ], feed_dict={x: batch_x, y: batch_y})
            validation_accuracy += (accuracy * len(batch_x))
            writer.add_summary(s, i)
        accuracy = validation_accuracy/num_valid_examples
        
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(accuracy))
        print()
        saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
        print("Model Checkpoint saved")

def make_hparam_string(learning_rate, use_drouput, dropout_rate, use_image_distortion):
    drop_param = "Dropout" if use_drouput else "no_dropout"
    image_distort_param = "with_image_distortion" if use_image_distortion else "no_image_distortion"
    return "lr_%.0E__%s__dropout_rate_%s__%s" % (learning_rate, drop_param, dropout_rate, image_distort_param)

# def testrun(X_train, y_train, X_valid, y_valid, X_test, y_test):
#     n_train, n_validation, n_test, image_shape, n_classes, class_file_df = data_summary()
#     drop_rate = 0
#     for learning_rate in [1E-3, 1E-4]:
#         hparam_string = make_hparam_string(learning_rate, False, False)
#         print('Starting run for %s' % hparam_string)
#         ModelArch(X_train, y_train, X_valid, y_valid, X_test, y_test,\
#                   learning_rate, n_classes, hparam_string, False, 0,\
#                   False, False)

def main():
      # You can try adding some more learning rates
    X_train, y_train, X_valid, y_valid, pX_test, y_test = load_pickled_data()
    n_train, n_validation, n_test, image_shape, n_classes, class_file_df = \
        data_summary(X_train, X_valid, pX_test)
    pX_train = nomalize_batch(images_to_gray(X_train))
    X_valid = nomalize_batch(images_to_gray(X_valid))
    dropout_rate = 0
    for learning_rate in [1E-3, 1E-4]:
        for image_distort_param in [False, True]:
            for drop_param in [False, True]:
                if image_distort_param:
                    X_train = [random_image_distortion(img) for img in X_train]
                else:
                    X_train = pX_train
                if drop_param:
                    for dropout_rate in [0.25, 0.5, 0.75]:
                        # Construct a hyperparameter string for each one
                        hparam_string = make_hparam_string(learning_rate, drop_param, dropout_rate, image_distort_param)
                        print('Starting run for %s' % hparam_string)
                        ModelArch(X_train, y_train, X_valid, y_valid, pX_test, y_test,\
                            learning_rate, n_classes, hparam_string, drop_param, dropout_rate,\
                            image_channels=1)
                else:
                    hparam_string = make_hparam_string(learning_rate, drop_param, image_distort_param)
                    print('Starting run for %s' % hparam_string)
                    ModelArch(X_train, y_train, X_valid, y_valid, pX_test, y_test,\
                        learning_rate, n_classes, hparam_string, drop_param, dropout_prob=0,\
                        image_channels=1)
    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)

if __name__ == '__main__':
    main()
