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
from tensorflow.keras.optimizers import Adam, Adagrad
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession #, RunOptions
import segmentation_models
import numpy as np
from os.path import basename
from utils.read_odom import denormalize, read_odom

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# run_opts = RunOptions(report_tensor_allocations_upon_oom = True)
            
def main(model_name, model, num_epochs, batch_size):
    '''Trains model.'''
    
    segmentation_models.set_framework('tf.keras')
    
    #Build list of training filenames
    X_folderpath=r"G:\Documents\KITTI\data\train\X\\"
    y_folderpath=r"G:\Documents\KITTI\data\train\y\\"
    X_filelist=glob(X_folderpath+'*.png')
    y_filelist=glob(y_folderpath+'*.png')
    
    #Build list of validation filenames
    X_val_folderpath=r"G:\Documents\KITTI\data\val\X\\"
    y_val_folderpath=r"G:\Documents\KITTI\data\val\y\\"
    X_val_filelist=glob(X_val_folderpath+'*.png')
    y_val_filelist=glob(y_val_folderpath+'*.png')
    
    model=model()
    
    #Define losses for each output
    losses = {'depth_output': 'mean_squared_error',
              "vo_output": 'mean_squared_logarithmic_error'} #mean_squared_logarithmic_error
    #model.load_weights(r"parallel_unets_with_tf_weights_best.hdf5")
    model.compile(loss=losses,optimizer=Adam(lr=5e-6)) #, options = run_opts) #1e-6
    #lr=5e-6
    
    #model.compile(loss=losses,optimizer=Adagrad(lr=5e-4))
    
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
               num_epochs=50,batch_size=4)
    show_test_image=True
    
    #Save model
    save_model(model,serialize_type='yaml',
                          model_name=f'{model_name}_kitti_model',
                          save_weights=False)
    
    save_model(model,serialize_type='json',
                          model_name=f'{model_name}_kitti_model',
                          save_weights=False)
    
    if show_test_image:
        #"G:\Documents\KITTI\data\train\X\2011_09_30_drive_0016_sync_0000000006.png"
        #G:\Documents\KITTI\data\val\X\2011_09_30_drive_0018_sync_0000000135.png"
        image1_path=r"G:\Documents\KITTI\data\train\X\2011_09_30_drive_0016_sync_0000000006.png"
        image2_path=r"G:\Documents\KITTI\data\train\X\2011_09_30_drive_0016_sync_0000000005.png"
        image1_odom=read_odom(sequence_id="2011_09_30_drive_0016",desired_frame=6)
        image2_odom=read_odom(sequence_id="2011_09_30_drive_0016",desired_frame=5)
        
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
        model.load_weights(f"{model_name}_weights_best.hdf5")
        y_est,odom_dt=model.predict([image1,image2])
        y_est=y_est.reshape((192,640))*255 #De-normalize for depth viewing
        #Save/view results
        heatmap(y_est,save=False,name=f'{image_name}_{model_name}_plasma',cmap='plasma')
        #odom_dt=denormalize(odom_dt.reshape(6))
        odom_dt=odom_dt.reshape(6)
        odom_dt=odom_dt.reshape((2,3))
        print('Predicted RPYXYZ:')
        print(odom_dt)
        
        odom_dt_actual=image1_odom-image2_odom
        odom_dt_actual=odom_dt_actual.reshape((2,3))
        print('Actual RPYXYZ:')
        print(odom_dt_actual)
        