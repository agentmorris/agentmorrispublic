#
# Code to render sample images and count annotations in the "Aerial Seabirds
# West Africa" dataset:
#
# https://lila.science/datasets/aerial-seabirds-west-africa/
#
# Annotations are points in a .csv file.
#

#%% Constants and imports

import pandas as pd
from tqdm import tqdm

from visualization import visualization_utils as visutils
from PIL import Image, ImageDraw

# PIL gets very sad when you try to load large images, suppress the error
Image.MAX_IMAGE_PIXELS = None

annotation_csv_file = r'gG:\temp\drone-datasets\aerial-seabirds-west-africa\labels_birds_full.csv'
data_file = r'g:\temp\drone-datasets\aerial-seabirds-west-africa\seabirds_rgb.tif'
output_file = r'g:\temp\aerial-seabirds-annotated.jpg'

# From gdalinfo
image_width_meters = 292.482 # 305.691
image_height_meters = 305.691 # 292.482


#%% Read and summarize annotations

df = pd.read_csv(annotation_csv_file)

species = set(df.label)
species_string = ''
for s in species:
    species_string += s.lower() + ','
species_string = species_string[0:-1]

print('Species present:')
print(species_string)

print('Number of annotations:')
print(len(df))
# import clipboard; clipboard.copy(species_string)


#%% Render points on the image (test)

if False:

    pil_im = visutils.open_image(data_file)
    print(pil_im.size)
    
    draw = ImageDraw.Draw(pil_im)
    
    x0 = 100
    x1 = 2000
    y0 = 100
    y1 = 2000
    
    draw.ellipse((x0,y0,x1,y1),fill=(255,0,0),outline=(0,0,0),width=100)
    pil_im.save(r'g:\temp\aerial-seabirds-test.jpg',quality=60)
    
    print('Done rendering')
    

#%% Render points on the image (for real)

pil_im = visutils.open_image(data_file)
print('Opened image with size: {}'.format(str(pil_im.size)))

draw = ImageDraw.Draw(pil_im)

ann_radius = 50
ann_radius = 2

# i_ann = 0; row = df.iloc[i_ann]
for i_ann,row in tqdm(df.iterrows(),total=len(df)):
    
    x_meters = row['X']
    y_meters = row['Y']
    x_relative = x_meters / image_width_meters
    y_relative = y_meters / image_height_meters
    
    x_pixels = x_relative * pil_im.size[0]
    y_pixels = y_relative * pil_im.size[1]
    y_pixels = pil_im.size[1] - y_pixels
    
    x0 = x_pixels - ann_radius
    y0 = y_pixels - ann_radius
    x1 = x_pixels + ann_radius
    y1 = y_pixels + ann_radius

    # draw.ellipse((x0,y0,x1,y1),fill=None,outline=(255,0,0),width=10)
    draw.ellipse((x0,y0,x1,y1),fill=(255,0,0),outline=None)
        
pil_im.save(output_file,quality=60)

print('Done rendering')    
