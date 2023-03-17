#
# usgs-geese-inference.py
#
# Run inference on a folder of images, by breaking each image into overlapping 
# 1280x1280 patches, running the model on each patch, and eliminating redundant
# boxes from the results.
#


#%% Constants and imports

import os
from visualization import visualization_utils as visutils

from detection import pytorch_detector

# We will explicitly verify that images are actually this size
expected_image_width = 8688
expected_image_height = 5792

patch_size = (1280,1280)


#%% Support functions

def get_patch_boundaries(image_size,patch_size,patch_stride=None):
        
    if patch_stride is None:
        patch_stride = (round(patch_size[0]/2),round(patch_size[1]/2))
        
    image_width = image_size[0]
    image_height = image_size[1]
        
    def add_patch_row(patch_start_positions,y_start):
        """
        Add one row to our list of patch start positions, i.e.
        loop over all columns.
        """
        x_start = 0; x_end = x_start + patch_size[0] - 1
        
        while(True):
            
            patch_start_positions.append([x_start,y_start])
            
            x_start += patch_stride[0]
            x_end = x_start + patch_size[0] - 1
             
            if x_end == image_width - 1:
                break
            elif x_end > (image_width - 1):
                overshoot = (x_end - image_width) + 1
                x_start -= overshoot
                x_end = x_start + patch_size[0] - 1
                patch_start_positions.append([x_start,y_start])
                break
        
        # ...for each column
        
        return patch_start_positions
        
    patch_start_positions = []
    
    y_start = 0; y_end = y_start + patch_size[1] - 1
        
    while(True):
    
        patch_start_positions = add_patch_row(patch_start_positions,y_start)
        
        y_start += patch_stride[1]
        y_end = y_start + patch_size[1] - 1
        
        if y_end == image_height - 1:
            break
        elif y_end > (image_height - 1):
            overshoot = (y_end - image_height) + 1
            y_start -= overshoot
            y_end = y_start + patch_size[1] - 1
            patch_start_positions = add_patch_row(patch_start_positions,y_start)
            break
    
    # ...for each row
    
    assert patch_start_positions[-1][0]+patch_size[0] == image_width
    assert patch_start_positions[-1][1]+patch_size[1] == image_height
    
    return patch_start_positions


#%%

def run_model_on_image(model,image_fn):
    
    #%%
    
    patch_stride = None
    pil_im = visutils.open_image(image_fn)
    assert pil_im.size[0] == expected_image_width
    assert pil_im.size[1] == expected_image_height
    
    image_width = pil_im.size[0]
    image_height = pil_im.size[1]
    image_size = (image_width,image_height)
    patch_start_positions = get_patch_boundaries(image_size,patch_size,patch_stride)
    
    
def load_model(model_fn):

    device = None
    model = pytorch_detector.PTDetector(model_fn,device)

    
#%% Interactive driver

if False:
    
    pass

    #%%
    
    image_fn = '/media/user/My Passport/2017-2019/01_JPGs/2018/Replicate_2018-10-20/CAM2/CAM21601.JPG'
    assert os.path.isfile(image_fn)
    