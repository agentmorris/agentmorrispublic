########
#
# usgs-geese-data-import.py
#
# * Checks internal consistency of the USGS brant survey data
#
# * Convert original annotations from CountThings format to COCO Camera Traps
#
# * Makes reasonable efforts to identify which images among the unnanotated
#   majority are suitable as hard negative examples.
#
########

#%% Imports and constants

import os
import json

import numpy as np
import pandas as pd

from tqdm import tqdm

from visualization import visualization_utils as visutils

output_json_file = os.path.expanduser('~/data/usgs_geese.json')

base_folder = '/media/user/My Passport/2017-2019'
image_folder = os.path.join(base_folder,'01_JPGs')
annotations_folder = os.path.join(base_folder,'03_Manually_corrected_annotations')

# Should we include annotations that don't have a label?  They're not really a "mystery",
# they're just annotations that began life as false positives, which were later removed.
include_mystery_annotations = False

# We are only keeping images with annotations, but some images end up only having annotations
# that we filter out ("mystery annotations").  Should we keep those?
include_images_with_only_removed_annotations = False

assert os.path.isdir(image_folder)
assert os.path.isdir(annotations_folder)

"""
In general, image paths look like:

2017-2019/01_JPGs/2017/Replicate_2017-09-30/Cam1/293A0006.JPG

Corresponding annotations paths look like:

2017-2019/03_Manually_corrected_annotations/2017/Replicate_2017-09-30/Cam1/293A0006_i0001.csv
2017-2019/03_Manually_corrected_annotations/2017/Replicate_2017-09-30/Cam1/293A0006_i0001.json

The .csv and .json files are redundant, though we will verify this later.

Only images with evidence of birds have been annotated, so, e.g., there is no csv/json pair for this image:
    
2017-2019/01_JPGs/2017/Replicate_2017-09-30/Cam1/293A0001.JPG

"""
pass


#%% Enumerate annotation files

import path_utils

annotation_files = path_utils.recursive_file_list(annotations_folder)
image_files_raw = path_utils.recursive_file_list(image_folder)

image_files = [fn for fn in image_files_raw if (fn.lower().endswith('.jpg') and 'lagoon' not in fn.lower())]
json_files = [fn for fn in annotation_files if fn.endswith('.json')]
csv_files = [fn for fn in annotation_files if fn.endswith('.csv')]

print('Found {} images, {} .json files, and {} .csv files'.format(
    len(image_files),len(json_files),len(csv_files)))

# In practice this is 101505 images, and 11069 .json/.csv pairs


#%% Match annotation files to image files

image_files_relative = [os.path.relpath(fn,image_folder) for fn in image_files]
json_files_relative = [os.path.relpath(fn,annotations_folder) for fn in json_files]
csv_files_relative = [os.path.relpath(fn,annotations_folder) for fn in csv_files]

json_files_relative_set = set(json_files_relative)
csv_files_relative_set = set(csv_files_relative)
image_files_relative_set = set(image_files_relative)

# Make sure the .json and .csv files match
assert len(json_files_relative) == len(csv_files_relative)
for fn in csv_files_relative:
    assert fn.replace('.csv','.json') in json_files_relative_set

csv_to_image = {}

# csv_fn = csv_files_relative[0] 
csv_files_without_images = []
for csv_fn in csv_files_relative:
    
    # E.g. '2017/Replicate_2017-09-30/Cam1/293A0014_i0009.csv'
    
    tokens = csv_fn.rsplit('_',1)
    assert len(tokens) == 2
    bn = tokens[0]
    expected_image_name = bn + '.JPG'
    if expected_image_name in image_files_relative_set:
        csv_to_image[csv_fn] = expected_image_name
    else:
        # print('No matching image for annotation {}'.format(csv_fn))
        csv_files_without_images.append(csv_fn)    
        
print('Missing images for {} of {} csv files'.format(
    len(csv_files_without_images),len(csv_files_relative)))

# We are missing images for 254 files.  Most appear to be just filename errors.  We should
# probably go back and match these to their images later, it will mess up eval a bit if we
# have non-empty images that we think are empty.

images_with_annotations = set(csv_to_image.values())
images_without_annotations = []
for fn in image_files_relative:
    if fn not in images_with_annotations:
        images_without_annotations.append(fn)
        
print('No annotations available for {} of {} images'.format(
    len(images_without_annotations),len(image_files_relative)))


if False:
    sample_image = os.path.join(image_folder,list(images_with_annotations)[0])
    path_utils.open_file(sample_image)


#%% Remove some candidates from the list of possible hard negatives

# Among the images without annotations, some are more likely than others
# to corresopnd to annotation files that just got dropped in the wrong 
# folder.  Pass those along to the training script.
images_that_might_not_be_empty = []

# csv_fn_relative = csv_files_without_images[0]
for csv_fn_relative in csv_files_without_images:
    if 'Cam1/CAM2' in csv_fn_relative:
        csv_fn_relative_corrected = csv_fn_relative.replace('Cam1/CAM2','Cam2/CAM2')
    else:
        continue
        
    expected_image_fn_relative = csv_fn_relative_corrected.rsplit('_',1)[0] + '.JPG'
    if os.path.isfile(os.path.join(image_folder,expected_image_fn_relative)):
        if expected_image_fn_relative in images_without_annotations:
            assert expected_image_fn_relative not in images_with_annotations
            images_that_might_not_be_empty.append(expected_image_fn_relative)

# This entire folder of annotations is missing
assert os.path.isdir(os.path.join(annotations_folder,'2019/Replicate_2019-10-09/Cam4 is missing'))
for fn in images_without_annotations:
    if '2019/Replicate_2019-10-09/Cam4' in fn:
        images_that_might_not_be_empty.append(fn)

manual_non_blanks = [
    '2017/Replicate_2017-10-01/Cam2/CAM28454.JPG',
    '2019/Replicate_2019-10-11/Cam3/CAM38014.JPG'        
]

manual_non_blank_images = []
for fn in manual_non_blanks:
    assert fn in images_without_annotations
    manual_non_blank_images.append(fn)
assert len(manual_non_blank_images) == len(manual_non_blanks)    

images_that_might_not_be_empty.extend(manual_non_blank_images)

# Find replicate folders in the images that don't exist in the annotations
#
# In practice there are two of these:
#
# {'2017/Replicate_2017-10-03', '2018/Replicate_2018-10-17'}

# fn = images_without_annotations[0]
all_annotation_replicate_folders = set()
for fn in csv_files_relative:
    replicate_folder = '/'.join(fn.split('/')[0:2])
    assert 'Replicate' in replicate_folder
    all_annotation_replicate_folders.add(replicate_folder)
    
image_replicate_folders_without_annotations = set()
for fn in images_without_annotations:
    replicate_folder = '/'.join(fn.split('/')[0:2])
    assert 'Replicate' in replicate_folder
    if replicate_folder not in all_annotation_replicate_folders:
        image_replicate_folders_without_annotations.add(replicate_folder)
        images_that_might_not_be_empty.append(fn)
    
print('Marked {} images as possibly-not-empty'.format(len(images_that_might_not_be_empty)))


#%% Build up COCO-formatted dataset

# Along the way, verify that the .csv files and .json files agree with each other

def isnan(v):
    if not isinstance(v,float):
        return False
    return np.isnan(v)

mystery_annotations = []
image_id_to_image = {}
empty_files = []
annotations = []

next_category_id = 0
category_name_to_category_id = {}

# i_file = 0; csv_fn_relative = csv_files_relative[i_file]
for i_file,csv_fn_relative in tqdm(enumerate(csv_files_relative),total=len(csv_files_relative)):
    
    if csv_fn_relative not in csv_to_image:
        continue
    
    im = {}
    
    json_fn_relative = csv_fn_relative.replace('.csv','.json')
    image_fn_relative = csv_to_image[csv_fn_relative]
    image_id = image_fn_relative.replace('/','_')
    
    # Some images had multiple identical annotation files (I don't know why)
    if image_id in image_id_to_image:
        
        previous_ann_file = image_id_to_image[image_id]['annotation_file']
        previous_ann_file_size = os.path.getsize(os.path.join(annotations_folder,previous_ann_file))
        
        ignore_ann_file = json_fn_relative
        ignore_ann_file_size = os.path.getsize(os.path.join(annotations_folder,ignore_ann_file))
        
        assert previous_ann_file_size == ignore_ann_file_size
        
        if False:
            print('Warning: multiple annotation files for image ID {}'.format(image_id))
            print('Previous annotation file was {} ({} bytes)'.format(
                previous_ann_file,previous_ann_file_size))
            print('Ignoring annotation file {} ({} bytes)'.format(
                ignore_ann_file,ignore_ann_file_size))
        continue
    
    csv_fn = os.path.join(annotations_folder,csv_fn_relative)
    json_fn = os.path.join(annotations_folder,json_fn_relative)
    image_fn = os.path.join(image_folder,image_fn_relative)
    
    pil_im = visutils.open_image(image_fn)
    image_w = pil_im.size[0]
    image_h = pil_im.size[1]
    
    im['width'] = image_w
    im['height'] = image_h
    im['annotation_file'] = json_fn_relative
    im['file_name'] = image_fn_relative
    im['id'] = image_id
    
    assert all([os.path.isfile(s) for s in [csv_fn,json_fn,image_fn]])    
    
    # Load both .json and .csv data
    with open(json_fn,'r') as f:
        json_data = json.load(f)
        
    if (os.path.getsize(csv_fn) <= 5):
        assert len(json_data) == 0
        empty_files.append(csv_fn)
        continue
    
    if (len(json_data) == 0):
        raise ValueError('Empty .json file with non-empty csv file')        
    
    df = pd.read_csv(csv_fn)
    assert len(df) == len(json_data)
    
    # Make sure they agree
    
    # For each row in the .csv file
    
    n_annotations_this_file = 0
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
    
        # The .json and .csv files should be ordered identically            
        json_sample = json_data[i_row]
        
        ann = {}
        ann['id'] = im['id'] + '_' + str(i_row).zfill(4)
        ann['image_id'] = im['id']
        
        if isnan(row['Label']):
            assert json_sample['Label'] == ''
            json_sample['filename'] = json_fn
            assert json_sample['ConfidenceLevel'] < 0
            mystery_annotations.append(json_sample)
            if not include_mystery_annotations:
                continue            
        
        w = json_sample['Width_Pixels']
        h = json_sample['Height_Pixels']
        x = json_sample['X'] - w/2.0
        y = json_sample['Y'] - h/2.0
        
        ann['bbox'] = [x,y,w,h]
                                
        # A json entry looks like:
        """
        {'Label': '46',
         'X': 5761.0,
         'Y': 4173.0,
         'ConfidenceLevel': 7,
         'Color': 1,
         'Radius_Pixels': 7,
         'Width_Pixels': 14,
         'Height_Pixels': 14,
         'Unit': 'Pixels',
         'SpeciesCategory': 'Brant'}
        """
        
        # These fields are identical across the .json/.csv files
        for s in ['ConfidenceLevel',
                  'Width_Pixels','Height_Pixels',
                  'Radius_Pixels','Unit']:
            assert json_sample[s] == row[s]
        
        assert json_sample['Unit'] == 'Pixels'
    
        # This field is identical, except that it's a string in the .json file
        if json_sample['Label'] != '':
            assert int(json_sample['Label']) == row['Label']
        
        # X/Y coordinates are similar, but not always exactly the same, across the
        # .json/.csv files
        for s in ['X','Y']:
            assert abs(json_sample[s] - row[s]) < 0.001
        
        # The 'mystery annotations' have 'Category' instead of 'SpeciesCategory'
        if not 'SpeciesCategory' in json_sample:
            
            assert 'Category' in json_sample
            assert json_sample['Label'] == '' and json_sample['Category'] == ''
        
        else:
            
            # For some reason, "SpeciesCategory" is just "Category" in some .csv files
            csv_category = ''
            if 'SpeciesCategory' in row:
                csv_category = row['SpeciesCategory']
            else:
                csv_category = row['Category']
            assert json_sample['SpeciesCategory'] == csv_category
        
        # These appear to vary from file to file, though I can't figure out 
        # whether they're really meaningful
        if False:
            assert json_sample['Radius_Pixels'] == 9 and \
                json_sample['Height_Pixels'] == 18 and \
                json_sample['Width_Pixels'] == 18
        
        if 'SpeciesCategory' in json_sample:
            category_name = json_sample['SpeciesCategory']        
        else:
            assert 'Category' in json_sample and json_sample['Category'] == ''
            category_name = 'Mystery'

        if category_name not in category_name_to_category_id:
            category_name_to_category_id[category_name] = next_category_id
            next_category_id += 1
        ann['category_id'] = category_name_to_category_id[category_name]
        
        annotations.append(ann)
        
        n_annotations_this_file += 1
                
    # ...for each annotation in this file
    
    if (n_annotations_this_file > 0) or (include_images_with_only_removed_annotations):
        image_id_to_image[image_id] = im
        
# ...for each file

##%% Write COCO .json file

images = list(image_id_to_image.values())

print('\nParsed {} annotations for {} images'.format(len(annotations),len(images)))

info = {}
info['version'] = '2023.03.10.00'
info['description'] = 'USGS brant survey data'

categories = []
for category_name in category_name_to_category_id:
    category_id = category_name_to_category_id[category_name]
    category = {
        'id':category_id,
        'name':category_name
        }
    categories.append(category)

d = {}
d['images'] = images
d['annotations'] = annotations
d['categories'] = categories
d['info'] = info 
d['images_without_annotations'] = images_without_annotations
d['images_that_might_not_be_empty'] = images_that_might_not_be_empty

with open(output_json_file,'w') as f:
    json.dump(d,f,indent=2)

print('Finished writing output to {}'.format(output_json_file))


#%% Scrap

if False:

    #%% Create temporary folder for output
    
    # Only used 
    output_dir = os.path.expanduser('~/data/usgs-geese-tmp')
    os.makedirs(output_dir,exist_ok=True)


    #%% Render bounding boxes for one image (independent of the .json data)
    
    i_file = 0
    csv_fn_relative = csv_files_relative[0]
    json_fn_relative = csv_fn_relative.replace('.csv','.json')
    image_fn_relative = csv_to_image[csv_fn_relative]
    
    json_fn = os.path.join(annotations_folder,json_fn_relative)
    image_fn = os.path.join(image_folder,image_fn_relative)
    
    with open(json_fn,'r') as f:
        json_data = json.load(f)
    
    image_relative_fn = os.path.relpath(image_fn,base_folder)
    image_preview_folder = os.path.join(output_dir,'image-preview')
    os.makedirs(image_preview_folder,exist_ok=True)
    output_file = os.path.join(image_preview_folder,image_relative_fn.replace('/','_') + \
                               '_preview.jpg')
    im = visutils.open_image(image_fn)
    image_w = im.size[0]
    image_h = im.size[1]
    
    detection_formatted_boxes = []
    
    category_name_to_category_id = {}
    next_category_id = 0
    
    # i_ann = 0; ann = json_data[i_ann]
    for i_ann,ann in enumerate(json_data):
        det = {}
        det['conf'] = 1.0
        box_center_x = ann['X'] / image_w
        box_center_y = ann['Y'] / image_h
        box_w = ann['Width_Pixels'] / image_w
        box_h = ann['Height_Pixels'] / image_h
        box_x_min = box_center_x - (box_w / 2)
        box_y_min = box_center_y - (box_h / 2)
        box = [box_x_min,box_y_min,box_w,box_h]
        det['bbox'] = box    
        
        if 'Category' in ann:
            assert 'SpeciesCategory' not in ann
            assert ann['Category'] == ''
            continue
        
        category_name = ann['SpeciesCategory']
        if category_name not in category_name_to_category_id:
            category_name_to_category_id[category_name] = next_category_id
            next_category_id += 1
        det['category'] = category_name_to_category_id[category_name]
        detection_formatted_boxes.append(det)
        
    category_id_to_name = {v:k for k,v in category_name_to_category_id.items()}
    visutils.draw_bounding_boxes_on_file(image_fn, output_file, detection_formatted_boxes,       
                                         confidence_threshold=0.0,
                                         detector_label_map=category_id_to_name)
    
    path_utils.open_file(output_file)
    
    
    #%% Read .json file and render one image
    
    from collections import defaultdict
    
    output_file = os.path.join(output_dir,'usgs_geese.json')
    with open(output_file,'r') as f:
        d = json.load(f)
    
    image_id_to_annotations = defaultdict(list)
    for ann in d['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
    
    i_image = 115
    im = d['images'][i_image]
    annotations = image_id_to_annotations[im['id']]
    print('Found {} annotations for this image'.format(len(annotations)))
    
    category_id_to_name = {c['id']:c['name'] for c in d['categories']}
    
    boxes = []
    categories = []
    for ann in annotations:
        boxes.append(ann['bbox'])
        categories.append(ann['category_id'])
        
    input_file = os.path.join(image_folder,im['file_name'])
    output_file = os.path.join(image_preview_folder,im['file_name'].replace('/','_') + \
                               '_preview.jpg')
    
    visutils.draw_db_boxes_on_file(input_file, output_file, 
                                   boxes=boxes, classes=categories,
                                   label_map=category_id_to_name, thickness=4, expansion=0)
    
    path_utils.open_file(output_file)
    
    
    #%% Check DB integrity
    
    from data_management.databases import integrity_check_json_db
    
    options = integrity_check_json_db.IntegrityCheckOptions()
    options.baseDir = image_folder
    options.bCheckImageSizes = False
    options.bCheckImageExistence = True
    options.bFindUnusedImages = False
    options.bRequireLocation = False
    
    sorted_categories, _, _= integrity_check_json_db.integrity_check_json_db(output_json_file, options)
    
    """
    424790 Brant
     47561 Canada
     41275 Other
      5631 Gull
      2013 Emperor
    """
    
    n_non_mystery_boxes = 0
    n_mystery_boxes = 0
    
    for c in sorted_categories:
        if c['name'] != 'Mystery':
            n_non_mystery_boxes += c['_count']
        else:
            n_mystery_boxes += c['_count']
    
    print('Mystery boxes: {}'.format(n_mystery_boxes))
    print('Non-mystery boxes: {}'.format(n_non_mystery_boxes))
    
    
    #%% Preview some images
    
    from visualization import visualize_db
    from path_utils import open_file
    
    viz_options = visualize_db.DbVizOptions()
    viz_options.num_to_visualize = 20
    viz_options.trim_to_images_with_bboxes = True
    viz_options.add_search_links = False
    viz_options.sort_by_filename = False
    viz_options.parallelize_rendering = True
    viz_options.include_filename_links = True
    
    html_output_file, _ = visualize_db.process_images(db_path=output_json_file,
                                                        output_dir=os.path.join(output_dir,'preview'),
                                                        image_base_dir=image_folder,
                                                        options=viz_options)
    open_file(html_output_file)
    
    
    #%% Explore .csv files that don't have matching images
    
    """
    There are 13,000 boxes with valid labels in 254 unmatched images.  Around 2887 of those
    boxes can be remapped with a simple path substitution, but it makes the code a lot more
    complicated, and that still leaves 10,000 just floating around, so I'm going to ignore
    those.  What I will do is just leave those off of the list of candidate hard negatives.
    """
    
    n_valid_labels = 0
    
    n_failed_remappings_with_valid_labels = 0
    
    n_valid_labels_in_remapped_images = 0
    n_valid_labels_in_unmapped_images = 0
    
    csv_to_replacement_image = {}
    
    # i_file = 0; csv_fn_relative = csv_files_without_images[i_file]
    for i_file,csv_fn_relative in tqdm(enumerate(csv_files_without_images),
                                       total=len(csv_files_without_images)):
        
        assert csv_fn_relative not in csv_to_image 
        json_fn_relative = csv_fn_relative.replace('.csv','.json')
        image_id = image_fn_relative.replace('/','_')
        
        csv_fn = os.path.join(annotations_folder,csv_fn_relative)
        json_fn = os.path.join(annotations_folder,json_fn_relative)
        
        assert os.path.isfile(csv_fn)
        assert os.path.isfile(json_fn)
        
        # E.g. '2017/Replicate_2017-09-30/Cam1/293A0014_i0009.csv'    
        tokens = csv_fn_relative.rsplit('_',1)
        assert len(tokens) == 2
        bn = tokens[0]
        expected_image_name = bn + '.JPG'
        assert expected_image_name not in image_files_relative_set
        
        with open(json_fn,'r') as f:
            json_labels = json.load(f)
        
        n_valid_labels_this_image = 0
        for label in json_labels:
            if label['Label'] != '' and label['ConfidenceLevel'] > 0:
                n_valid_labels_this_image += 1
        n_valid_labels += n_valid_labels_this_image
        
        if (n_valid_labels_this_image > 0):
            b_remapped = False
            if 'Cam1/CAM2' in csv_fn_relative:
                csv_fn_relative_corrected = csv_fn_relative.replace('Cam1/CAM2','Cam2/CAM2')
                expected_image_fn_relative = csv_fn_relative_corrected.rsplit('_',1)[0] + '.JPG'
                if os.path.isfile(os.path.join(image_folder,expected_image_fn_relative)):
                    csv_to_replacement_image[csv_fn_relative] = expected_image_fn_relative
                    n_valid_labels_in_remapped_images += n_valid_labels_this_image
                    b_remapped = True
            if not b_remapped:
                n_failed_remappings_with_valid_labels += 1
                n_valid_labels_in_unmapped_images += n_valid_labels_this_image
                
    # ...for each unmapped .csv file
                
    print('\nTotal of {} valid labels in {} annotation files without images'.format(
        n_valid_labels,len(csv_files_without_images)))
