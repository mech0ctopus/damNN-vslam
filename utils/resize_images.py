#Utility functions
from PIL import Image
from glob import glob
from os.path import basename, exists
from os import mkdir

def resize_image(input_image_path,output_image_path,size=(640,192)):
    '''Resize image then save.'''
    #read image
    input_image=Image.open(input_image_path)
    #resize image
    element_image=input_image.resize((size[0],size[1]),Image.ANTIALIAS)
    #save in output folder
    element_image.save(output_image_path,quality=95) #95 is best

def resize_images(input_image_folderpath,size=(640,192)):
	'''Resize all images in folder then save.'''
	#Append folderpath if needed
	if input_image_folderpath.endswith('\\')==False:
		input_image_folderpath=str(input_image_folderpath)+ '\\'
	output_image_folderpath=input_image_folderpath+'resize'+'\\'
	#Make output folder if it doesn't exist
	if exists(output_image_folderpath)==False:
		mkdir(output_image_folderpath)
	#Build list of input images
	input_images=glob(input_image_folderpath+'*'+'.png')
	num_images=len(input_images)
	for idx, input_image in enumerate(input_images):
		out_filename=output_image_folderpath+basename(input_image).split('.')[0]+'.png'
		print('Resizing '+ str(idx+1)+'/'+str(num_images))
		resize_image(input_image,out_filename,size=(size[0],size[1]))
 
if __name__ == '__main__':
    input_image_folderpath = r"G:\Documents\KITTI\raw_data\Depth\2011_09_30_drive_0028_sync\colorized"
    resize_images(input_image_folderpath, size=(640,192))