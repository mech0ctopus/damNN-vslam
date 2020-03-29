# Kyle J. Cantrell, Craig D. Miller, and Brian Slagowski
# kjcantrell@wpi.edu, cdmiller@wpi.edu, and bslagowski@wpi.edu
# Advanced Robot Navigation
#
# Dense Accurate Map-building using Neural Networks

from glob import glob
from utils.deep_utils import save_model
from models import models
from models.generators import _batchGenerator, _valBatchGenerator
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import datetime
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import segmentation_models

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
            
def main(model_name, model, num_epochs, batch_size):
    '''Trains model.'''
    
    segmentation_models.set_framework('tf.keras')
    print(segmentation_models.framework())
    
    #Build list of training filenames
    X_folderpath=r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Data\Train\X_rgb\\"
    y_folderpath=r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Data\Train\y_depth\\"
    X_filelist=glob(X_folderpath+'*.png')
    y_filelist=glob(y_folderpath+'*.png')
    
    #Build list of validation filenames
    X_val_folderpath=r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Data\Val\X_rgb\\"
    y_val_folderpath=r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Data\Val\y_depth\\"
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
    model=models.unet
    model_name='unet'
    model=main(model_name=model_name,model=model,
               num_epochs=2,batch_size=2)
    
    #Save model
    save_model(model,serialize_type='yaml',
                          model_name=f'{model_name}_nyu_model',
                          save_weights=False)
    
    save_model(model,serialize_type='json',
                          model_name=f'{model_name}_nyu_model',
                          save_weights=False)