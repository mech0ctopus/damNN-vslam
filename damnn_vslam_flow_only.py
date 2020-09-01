# Dense Accurate Map-building using Neural Networks

from glob import glob
from utils.deep_utils import rgb_read
from models import models
from models.flow_generators import _batchGenerator, _valBatchGenerator
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
    X_folderpath=r"data\train\flow\\"
    X_filelist=glob(X_folderpath+'*.png')
    
    #Build list of validation filenames
    X_val_folderpath=r"data\val\flow\\"
    X_val_filelist=glob(X_val_folderpath+'*.png')
    
    model=model()
    losses={'rpy_output':'msle',
            'xyz_output':'msle'}
    
    # model.compile(loss=losses,optimizer=Adagrad(0.001))
    model.compile(loss=losses,optimizer=Adam(1e-7)) #Val_loss:21.76883 with LR=0.001
    
    #Save best model weights checkpoint
    filepath=f"{model_name}_weights_best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min')
    # filepath2=f"{model_name}_weights_best_trainingloss.hdf5"
    # checkpoint2 = ModelCheckpoint(filepath2, monitor='loss', verbose=1, 
    #                               save_best_only=True, mode='min')
    
    #Tensorboard setup
    log_dir = f"logs\\{model_name}\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")        
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    
    callbacks_list = [checkpoint, tensorboard_callback] #checkpoint2, 
    
    model.fit_generator(_batchGenerator(X_filelist,batch_size),
                        epochs=num_epochs,
                        steps_per_epoch=len(X_filelist)//batch_size,
                        validation_data=_valBatchGenerator(X_val_filelist,batch_size),
                        validation_steps=len(X_val_filelist)//batch_size,
                        #validation_freq=1,
                        max_queue_size=1,
                        callbacks=callbacks_list,
                        verbose=2)
    
    return model
    
if __name__=='__main__':
    model=models.vo_from_flow
    model_name='vo_from_flow'
    model=main(model_name=model_name,model=model,
               num_epochs=200,batch_size=1)
        