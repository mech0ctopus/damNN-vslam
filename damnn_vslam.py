# Kyle J. Cantrell, Craig D. Miller, and Brian Slagowski
# kjcantrell@wpi.edu, cdmiller@wpi.edu, and bslagowski@wpi.edu
# Advanced Robot Navigation
#
# Dense Accurate Map-building using Neural Networks

from glob import glob
from utils.deep_utils import save_model, rgb_read, heatmap
from models import models
from models.generators import _batchGenerator, _valBatchGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import segmentation_models
import numpy as np
from os.path import basename

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
            
def main(model_name, model, num_epochs, batch_size):
    '''Trains model.'''
    
    segmentation_models.set_framework('tf.keras')
    print(segmentation_models.framework())
    
    #Build list of training filenames
    X_folderpath=r"images\training\X\\"
    y_folderpath=r"images\training\y\colorized\\"
    X_filelist=glob(X_folderpath+'*.png')
    y_filelist=glob(y_folderpath+'*.png')
    
    #Build list of validation filenames
    X_val_folderpath=r"images\testing\X\\"
    y_val_folderpath=r"images\testing\y\colorized\\"
    X_val_filelist=glob(X_val_folderpath+'*.png')
    y_val_filelist=glob(y_val_folderpath+'*.png')
    
    model=model()
    model.compile(loss='mean_squared_error',optimizer=Adam(lr=1e-5)) 

    #Save best model weights checkpoint
    filepath=f"{model_name}_weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    
    #Tensorboard setup
    log_dir = f"logs\\{model_name}\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")        
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    
    callbacks_list = [checkpoint, tensorboard_callback]
    
    model.fit_generator(_batchGenerator(X_filelist,y_filelist,batch_size),
                        epochs=num_epochs,
                        steps_per_epoch=len(X_filelist)//batch_size,
                        validation_data=_valBatchGenerator(X_val_filelist,y_val_filelist,batch_size),
                        validation_steps=len(X_val_filelist)//batch_size,
                        max_queue_size=1,
                        callbacks=callbacks_list,
                        verbose=2)
    
    return model
    
if __name__=='__main__':
    model=models.parallel_unets
    model_name='parallel_unets'
    model=main(model_name=model_name,model=model,
               num_epochs=2,batch_size=2)
    show_test_image=True
    
    #Save model
    save_model(model,serialize_type='yaml',
                          model_name=f'{model_name}_kitti_model',
                          save_weights=False)
    
    save_model(model,serialize_type='json',
                          model_name=f'{model_name}_kitti_model',
                          save_weights=False)
    
    if show_test_image==True:
        image1_path=r"images\testing\X\c_depth_0.PNG"
        #Read test image
        image1=rgb_read(image1_path) #640x480, 1242x375
        image1=image1.reshape(1,375,1242,3)
        image1=np.divide(image1,255).astype(np.float16)

        image2_path=r"images\testing\X\c_depth_1.PNG"
        #Read test image
        image2=rgb_read(image2_path) #640x480
        image2=image2.reshape(1,375,1242,3)
        image2=np.divide(image2,255).astype(np.float16)
        
        image_name=basename(image1_path).split('.')[0]
        #Predict depth
        y_est=model.predict([image1,image2])
        y_est=y_est.reshape((375,1242))*255 #De-normalize for depth viewing
        #Save results
        heatmap(y_est,save=False,name=f'{image_name}_{model_name}_plasma',cmap='plasma')
        