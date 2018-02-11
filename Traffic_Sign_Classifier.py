#%%
# Load pickled data
import pickle
def load_datasets():
    print("loading Datasets...")
    training_file = "./traffic-sign-data/train.p"
    validation_file= "./traffic-sign-data/valid.p"
    testing_file = "./traffic-sign-data/test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    print("Done.")
    return train,valid,test
train, valid, test = load_datasets()    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#%%
import numpy as np
import pandas as pd

class_file_name = "./signnames.csv"

class_file_df = pd.read_csv(class_file_name, delimiter=',')

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

#%%
#Let's look at what one of these pictures looks like
import random
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
# %matplotlib inline
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()
print(y_train[index])
fig = plt.figure(figsize=(2,2))
ax = fig.add_subplot(111)
ax.grid(b=False)
plt.imshow(image)
print(tf.summary.image('images', X_train))

#%%
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

#%%
#Image Preprocessing Helper Functions
import cv2
def image_to_YUV(img):
    """Applies the YUV transform"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def normalize(channels):
    """Normalizes RGB channels"""
    return [(channel-128)/128 for channel in channels]

#%%
# Get a copy of our test image and run some pre-processing on it to
# see how it looks
# image_copy = np.copy(image)
# img_yuv = gaussian_blur(img_yuv, 1)
# channels  = cv2.split(image_copy)
# # normalize the RGB color lines
# channels = normalize(channels)
# image_copy = cv2.merge(channels)
# # convert to yuv color space
# # img_yuv = image_to_YUV(image_copy)
# low_threshold = 1
# high_threshold = 10
# edges = cv2.Canny(channels[0], low_threshold, high_threshold)
# split out the channels
# channels  = cv2.split(img_yuv)
# merge them back together
# img_yuv_normal = cv2.merge(channels)
# img_yuv_normal = gaussian_blur(img_yuv_normal, 1)

# plt.subplot(1,2,1),plt.imshow(image)
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(img_yuv)
# plt.title('YUV Image'), plt.xticks([]), plt.yticks([])

# plt.show()

#%%
# looks good, let's put that into a function to be used later:
def preprocess_img(img):
    image_copy = np.copy(img)
    channels  = cv2.split(image_copy)
    image_copy = normalize(channels)
    return gaussian_blur(image_to_YUV(image_copy), 1)
    
#%%
# preprocess the images
# print("Pre-Processing....")
# for i in range(len(X_train)):
#     X_train[i] = preprocess_img(X_train[i])
# for i in range(len(X_valid)):
#     X_valid[i] = preprocess_img(X_valid[i])
# for i in range(len(X_test)):
#     X_test[i] = preprocess_img(X_test[i])
# print("Done")

#%%
#suffle our training data
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)
print("shuffled")

#%%
# set up the LeNetish Model
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def conv2d(x,W,b,strides=1):
    x = tf.nn.conv2d(x,W, strides=[1,strides,strides,1], padding="SAME")
    return tf.nn.bias_add(x,b)

def LeNetish(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    filter_heights = [5,5]
    filter_widths = [5,5]
    in_channels = [3,6]
    out_channels = [6,16]
    weights = {
        'con_layer1': tf.Variable(tf.truncated_normal([filter_heights[0],filter_widths[0],in_channels[0],out_channels[0]],mu,sigma)),
        'con_layer2': tf.Variable(tf.truncated_normal([filter_heights[1],filter_widths[1],in_channels[1],out_channels[1]]),mu,sigma),
        'fc_layer1': tf.Variable(tf.truncated_normal([5*5*16,120]),mu,sigma),
        'fc_layer2': tf.Variable(tf.truncated_normal([120,84]),mu,sigma),
        'logits': tf.Variable(tf.truncated_normal([84,n_classes]),mu,sigma)
    }
    biases = {
        'con_layer1': tf.Variable(tf.truncated_normal([6])),
        'con_layer2': tf.Variable(tf.truncated_normal([16])),
        'fc_layer1': tf.Variable(tf.truncated_normal([120])),
        'fc_layer2': tf.Variable(tf.truncated_normal([84])),
        'logits': tf.Variable(tf.truncated_normal([n_classes])),
    }
    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = tf.nn.conv2d(x, weights['con_layer1'], strides=[1,1,1,1], padding="VALID")

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
    
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = tf.nn.conv2d(conv1, weights['con_layer2'], strides=[1,1,1,1], padding="VALID")
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")

    # Flatten. Input = 5x5x16. Output = 400.
    conv2 = flatten(conv2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1 = tf.reshape(conv2, [-1, weights['fc_layer1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['fc_layer1']), biases['fc_layer1'])
    
    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2 = tf.reshape(fc1, [-1, weights['fc_layer2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['fc_layer2']), biases['fc_layer2'])
    
    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 10.
    logits = tf.add(tf.matmul(fc2, weights['logits']), biases['logits'])
    
    return logits

#%%
# setup the feautures, labels and trianing pipeline
EPOCHS = 10
BATCH_SIZE = 128
learn_rate = 0.001

# x is a placeholder for a batch of input images. y is a placeholder for a batch of output labels.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y,n_classes)

logits = LeNetish(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learn_rate)
training_operation = optimizer.minimize(loss_operation)

#%%
# Setup model evaluation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

#%%
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")