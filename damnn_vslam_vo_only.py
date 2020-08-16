# Dense Accurate Map-building using Neural Networks

from glob import glob
from utils.deep_utils import rgb_read
from models import models
from models.vo_generators import _batchGenerator, _valBatchGenerator
from models.losses import undeepvo_rpy_mse, undeepvo_xyz_mse, deepvo_mse
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adagrad, Adam
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import segmentation_models
import numpy as np
from os.path import basename
from utils.read_odom import read_odom

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# run_opts = RunOptions(report_tensor_allocations_upon_oom = True)
            
def main(model_name, model, num_epochs, batch_size):
    '''Trains model.'''
    
    segmentation_models.set_framework('tf.keras')
    
    #Build list of training filenames
    X_folderpath=r"data\train\X\\"
    y_folderpath=r"data\train\y\\"
    X_filelist=glob(X_folderpath+'*.png')
    y_filelist=glob(y_folderpath+'*.png')
    
    #Build list of validation filenames
    X_val_folderpath=r"data\val\X\\"
    y_val_folderpath=r"data\val\y\\"
    X_val_filelist=glob(X_val_folderpath+'*.png')
    y_val_filelist=glob(y_val_folderpath+'*.png')
    
    model=model()
    losses={'rpy_output':undeepvo_rpy_mse,
            'xyz_output':undeepvo_xyz_mse}
    
    #DeepVO uses Adagrad(0.001)
    # model.load_weights(r"C:\Users\craig\Documents\GitHub\damNN-vslam\Weights\20200714-191625_mock_undeepvo_weights_best.hdf5")
    model.compile(loss=losses,optimizer=Adam(1e-4,beta_2=0.99)) #UnDeepVO uses beta_2=0.99
    #model.compile(loss=deepvo_mse,optimizer=Adam(0.005,beta_2=0.99))
    
    #Save best model weights checkpoint
    filepath=f"{model_name}_weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    filepath2=f"{model_name}_weights_best_trainingloss.hdf5"
    checkpoint2 = ModelCheckpoint(filepath2, monitor='loss', verbose=1, 
                                 save_best_only=True, mode='min')
    
    #Tensorboard setup
    log_dir = f"logs\\{model_name}\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")        
    tensorboard_callback = TensorBoard(log_dir=log_dir, write_images=True)
    
    callbacks_list = [checkpoint, checkpoint2, tensorboard_callback]
    
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
    model=models.mock_espvo
    model_name='mock_espvo'
    model=main(model_name=model_name,model=model,
               num_epochs=50,batch_size=1)
    show_test_image=True
    
    if show_test_image:
        #"data\train\X\2011_09_30_drive_0016_sync_0000000006.png"
        #"data\val\X\2011_09_30_drive_0018_sync_0000000135.png"
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
        model.load_weights(f"{model_name}_weights_best.hdf5")
        rpyxyz_est=model.predict([image1,image2])

        print('Predicted RPYXYZ:')
        print(rpyxyz_est)
        
        odom_dt_actual=image1_odom-image2_odom
        odom_dt_actual=odom_dt_actual.reshape((2,3))
        print('Actual RPYXYZ:')
        print(odom_dt_actual)
        
