# -*- coding: utf-8 -*-
"""
https://github.com/jacobgil/keras-filter-visualization/issues/1
"""

from tensorflow.keras import backend as K
from models import models
from utils.read_odom import read_odom
from utils.deep_utils import rgb_read
import tensorflow as tf
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

tf.compat.v1.disable_eager_execution()

#Load images
image1_path=r"data\val\X\2011_09_30_drive_0018_sync_0000000135.png"
image2_path=r"data\val\X\2011_09_30_drive_0018_sync_0000000134.png"
image1_odom=read_odom(sequence_id="2011_09_30_drive_0018",desired_frame=135)
image2_odom=read_odom(sequence_id="2011_09_30_drive_0018",desired_frame=134)

#Read test image
image1=rgb_read(image1_path) #640x480, 1242x375
image1=image1.reshape(1,192,640,3)
image1=np.divide(image1,255).astype(np.float16)
#Read test image
image2=rgb_read(image2_path) #640x480
image2=image2.reshape(1,192,640,3)
image2=np.divide(image2,255).astype(np.float16)


#K.set_learning_phase(1)
#Configuration:
img_width, img_height = 640, 192
input_shape = (img_width, img_height, 1)
num_filters = 25
iterations = 20
weights_path = r"C:\Users\craig\Documents\GitHub\damNN-vslam\mock_undeepvo_weights_best.hdf5"
#layer_names = ['conv2d_1','conv2d_2']
#img_path = r'mnist_in.png'
img_path=None
filter_indexes = range(0, num_filters)

def save_filters(filters, img_width, img_height,layer_name):
    margin = 5
    n = int(len(filters)**0.5)
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index < len(filters):
                img = filters[i * n + j]
                img=img.reshape(640,192,3)
                stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                                 (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    cv2.imwrite('undeepvo_'+f'{layer_name}'+'_stitched_filters_%dx%d.png' % (n, n), stitched_filters)

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def get_output_layer(model, layer_name):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    #layer_dict = dict([(layer.name, layer) for layer in model.layers])
    #layer_output = layer_dict[layer_name].output
    layer_output = model.get_layer(layer_name).output
    return layer_output

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

#Define regularizations:
def blur_regularization(img, grads, size = (3, 3)):
    return cv2.blur(img, size)

def decay_regularization(img, grads, decay = 0.8):
    return decay * img

def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped

def gradient_ascent_iteration(loss_function, img, lr=0.9):
    loss_value, grads_value = loss_function([img])    
    gradient_ascent_step = img + grads_value * lr

    #Convert to row major format for using opencv routines
    grads_row_major = grads_value[0, :]
    img_row_major = gradient_ascent_step[0, :]

    #List of regularization functions to use
    regularizations = [blur_regularization, decay_regularization, clip_weak_pixel_regularization]

    #The reguarlization weights
    weights = np.float32([3, 3, 1])
    weights /= np.sum(weights)

    images = [reg_func(img_row_major, grads_row_major) for reg_func in regularizations]

    for idx, image in enumerate(images):
        image=image.reshape((192,640))
        images[idx]=image
    weighted_images = np.float32([w * image for w, image in zip(weights, images)])
    img = np.sum(weighted_images, axis = 0)

    #Convert image back to 1 x 3 x height x width
    img = np.float32([img])
    #print(img.shape)
    img=img.reshape(1,192,640,1)
    return img

def visualize_filter(input_img, filter_index, img_placeholder, layer, number_of_iterations = 20):
    loss = K.mean(layer[:, :, :, filter_index])
    grads = K.gradients(loss, img_placeholder)[0]
    grads = normalize(grads)
    # this function returns the loss and grads given the input picture
    iterate = K.function([img_placeholder], [loss, grads])

    img = input_img * 1

    # we run gradient ascent for 20 steps
    for i in range(number_of_iterations):
        img = gradient_ascent_iteration(iterate, img)

    # decode the resulting input image
    img = deprocess_image(img[0])
    #print("Done with filter", filter_index)
    return img

model=models.mock_undeepvo()
layer_names=[layer.name for layer in model.layers]
print(layer_names)
layer_names=[layer_names[3]]
#model.summary()
model.load_weights(weights_path)
input_placeholder = model.input
for layer_name in layer_names:
    # if 'conv' in layer_name:
    print(layer_name)
    layer = get_output_layer(model, layer_name)
    #print(model.input)
    if img_path is None:
        # we start from a gray image with some random noise
        init_img = np.random.random((1, img_width, img_height, 1)) * 20 + 28.
        print(init_img.shape)
    else:
        #img = cv2.imread(img_path, 1)
        img=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_width, img_height))
        init_img = img.reshape(1,192,640,1)
        #print(init_img.shape)
    
    vizualizations = [None] * len(filter_indexes)
    for i in tqdm(range(len(filter_indexes))):
        #for i,  in enumerate(filter_indexes):
        index = filter_indexes[i]
        vizualizations[i] = visualize_filter(init_img, index, input_placeholder,layer, iterations)
        #Save the visualizations see the progress made so far
        save_filters(vizualizations, img_width, img_height, layer_name)
print('Done.')