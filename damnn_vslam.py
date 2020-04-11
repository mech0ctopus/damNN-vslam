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
from tensorflow.compat.v1 import InteractiveSession, RunOptions
import segmentation_models
import numpy as np
from os.path import basename

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
#run_opts = RunOptions(report_tensor_allocations_upon_oom = True)
            
def main(model_name, model, num_epochs, batch_size):
    '''Trains model.'''
    
    segmentation_models.set_framework('tf.keras')
    print(segmentation_models.framework())
    
    #Build list of training filenames
    X_folderpath=r"G:\Documents\KITTI\raw_data\RGB\2011_10_03_drive_0042_sync\resize\\"
    y_folderpath=r"G:\Documents\KITTI\raw_data\Depth\2011_10_03_drive_0042_sync\colorized\resize\\"
    X_filelist=glob(X_folderpath+'*.png')
    y_filelist=glob(y_folderpath+'*.png')
    
    #Build list of validation filenames
    X_val_folderpath=r"G:\Documents\KITTI\raw_data\RGB\2011_09_30_drive_0016_sync\resize\\"
    y_val_folderpath=r"G:\Documents\KITTI\raw_data\Depth\2011_09_30_drive_0016_sync\colorized\resize\\"
    X_val_filelist=glob(X_val_folderpath+'*.png')
    y_val_filelist=glob(y_val_folderpath+'*.png')
    
    model=model()
    
    losses = {"depth_output": 'mean_squared_error',
              "vo_output": 'mean_squared_error'}

    model.compile(loss=losses,optimizer=Adam(lr=1e-5))#,options=run_opts) 
    
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
                        #validation_freq=1,
                        max_queue_size=1,
                        callbacks=callbacks_list,
                        verbose=2)
    
    return model
    
if __name__=='__main__':
    model=models.parallel_unets_with_tf
    model_name='parallel_unets_with_tf'
    model=main(model_name=model_name,model=model,
               num_epochs=5,batch_size=1) #2 (1:OOM)
    show_test_image=True
    
    #Save model
    save_model(model,serialize_type='yaml',
                          model_name=f'{model_name}_kitti_model',
                          save_weights=False)
    
    save_model(model,serialize_type='json',
                          model_name=f'{model_name}_kitti_model',
                          save_weights=False)
    
    if show_test_image:
        image1_path=r"images\test\X\0000000080.PNG"
        #Read test image
        image1=rgb_read(image1_path) #640x480, 1242x375
        image1=image1.reshape(1,192,640,3)
        image1=np.divide(image1,255).astype(np.float16)

        image2_path=r"images\test\X\0000000079.PNG"
        #Read test image
        image2=rgb_read(image2_path) #640x480
        image2=image2.reshape(1,192,640,3)
        image2=np.divide(image2,255).astype(np.float16)
        
        image_name=basename(image1_path).split('.')[0]
        #Predict depth
        y_est,odom_dt=model.predict([image1,image2])
        y_est=y_est.reshape((192,640))*255 #De-normalize for depth viewing
        #Save results
        heatmap(y_est,save=False,name=f'{image_name}_{model_name}_plasma',cmap='plasma')
        print(odom_dt)
        