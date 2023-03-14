#
# deepforest-test.py
# 
# A test script to run DeepForest on a couple of orthomosaics, and export the results to
# shapefiles.
#

#%% Constants and imports

import os

import matplotlib.pyplot as plt
from deepforest import main as deepforest_main
from deepforest import utilities as deepforest_utils

# The patch size matters a *lot*; I'm choosing a patch size that gets good results for the 
# small trees, and not worrying about the big trees.  It it quick to try different patch sizes,
# so it will likely make sense to tune this for each image.
patch_size = 400
patch_overlap = 0.5
iou_threshold = 0.01

# I'll process all .tif files in this folder
base_folder = '/home/user/data/deepforest-test'


#%% Convert files to RGB

# DeepForest does not like RGBA images, so I dropped the alpha channel:

"""
gdal_translate -b 1 -b 2 -b 3 Beanjavilo_Orthomosaic_export_FriFeb10094608634979.tif Beanjavilo_Orthomosaic_export_FriFeb10094608634979.rgb.tif 

gdal_translate -b 1 -b 2 -b 3 SoahanyBenoabo_Orthomosaic_export_ThuFeb09133938325845.tif SoahanyBenoabo_Orthomosaic_export_ThuFeb09133938325845.rgb.tif 
"""

#%% Enumerate files

filenames = os.listdir(base_folder)
filenames = [fn for fn in filenames if (fn.lower().endswith('.tif') or fn.lower().endswith('.tiff'))]
filenames = [os.path.join(base_folder,fn) for fn in filenames]

print('Enumerated {} .tif files:'.format(len(filenames)))
for fn in filenames:
    print(fn)


#%% DeepForest initialization

model = deepforest_main.deepforest()
model.use_release()
 

#%% Run DeepForest on each file, export to .csv and .shp

# fn = filenames[0]
for fn in filenames:
    
    results = model.predict_tile(raster_path = fn,
                                 patch_size = patch_size,
                                 patch_overlap = patch_overlap,
                                 iou_threshold = iou_threshold,
                                 return_plot = False)
    results_fn = fn + '_results_{}_{}_{}.csv'.format(patch_size,patch_overlap,iou_threshold)
    results.to_csv(results_fn,index=False)
    
    results_transformed = deepforest_utils.boxes_to_shapefile(results,base_folder,projected=True)
    shapefile_fn = fn + '_results_{}_{}_{}.shp'.format(patch_size,patch_overlap,iou_threshold)
    results_transformed.to_file(shapefile_fn)
