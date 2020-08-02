# visualize feature maps output from each block in the vgg model
from tensorflow.keras.models import Model
from matplotlib import pyplot
from models import models
from utils.deep_utils import rgb_read
from utils.read_odom import read_odom
import numpy as np
from os.path import basename

# load the model
model = models.mock_undeepvo()

# output all layers
ixs = [4, 5, 6, 7, 8, 9]

# model.layers[3] #Sequential
# model.layers[3] #Sequential
# model.layers[3] #Sequential
# model.layers[3] #Sequential
# model.layers[3] #Sequential
print(len(model.layers))
print(model.layers)
outputs=[model.layers[i].output for i in ixs]
# high_level_layers=[model.layers[i].output for i in ixs]

# outputs=[]
# for high_level_layer in high_level_layers:
#     for low_level_layer in high_level_layer.layers:
#         outputs.append(low_level_layer.output)
        
model = Model(inputs=model.inputs, outputs=outputs)

# load the image with the required shape
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

image_name=basename(image1_path).split('.')[0]
#Make sure we have best weights
#Predict depth and [RPY,XYZ]
model.load_weights(r"mock_undeepvo_weights_best.hdf5")

# get feature map for first hidden layer
feature_maps = model.predict([image1,image2])
# plot the output from each block
square = 8
for fmap in feature_maps:
    # plot all 64 maps in an 8x8 squares
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            print(fmap.shape)
            # plot filter channel in grayscale
            pyplot.imshow(fmap[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()
