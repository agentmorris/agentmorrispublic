
#!/usr/bin/env python3
"""
Convert Savmap Hugging Face dataset from parquet format to COCO format.

This script reads parquet files from the Hugging Face dataset and converts them
to COCO format with separate image files and a JSON annotation file.

Requirements:
    pip install pandas pyarrow pillow ipython megadetector-utils

Usage:
    python convert-savmap-huggingface.py

Input: /mnt/i/data/drone-data/reinhard-savmap/savmap-huggingface/data/train-*.parquet
Output: /mnt/i/data/drone-data/reinhard-savmap/savmap-huggingface/converted-to-coco/
"""

#%% Imports and constants

import json
import os
import io

from pathlib import Path
from typing import Dict, List, Any
from PIL import Image

import pandas as pd

parquet_input_folder = "/mnt/i/data/drone-data/reinhard-savmap/savmap-huggingface/data"
coco_output_folder = "/mnt/i/data/drone-data/reinhard-savmap/savmap-huggingface/converted-to-coco"
coco_json_path = os.path.join(coco_output_folder,"annotations.json")
image_output_folder = os.path.join(coco_output_folder,"images")


#%% Conversion function

def convert_savmap_to_coco(
    input_dir=parquet_input_folder,
    output_dir=coco_output_folder):
    """
    Convert Savmap dataset from parquet format to COCO format.

    Args:
        input_dir (str, optional): Directory containing parquet files
        output_dir (str, optional): Directory to write images and COCO JSON
    """

    assert os.path.isdir(input_dir), \
        'Folder {} does not exist'.format(input_dir)
    os.makedirs(output_dir,exist_ok=True)
    os.makedirs(image_output_folder,exist_ok=True)

    # Create output directories
    output_path = Path(output_dir)

    # Initialize COCO structure
    coco_data: Dict[str, Any] = {
        "info": {
            "description": "Savmap Dataset - Converted from Hugging Face",
            "url": "https://huggingface.co/datasets/fadel841/savmap",
            "version": "1.0",
            "year": 2025,
            "contributor": "fadel841",
            "date_created": "2025-11-10"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "animal",  # Default category name, can be updated if known
                "supercategory": "none"
            }
        ]
    }

    # Process all parquet files
    parquet_files = sorted(Path(input_dir).glob("train-*.parquet"))
    print(f"Found {len(parquet_files)} parquet files to process")

    image_id = 1
    annotation_id = 1

    for parquet_file in parquet_files:

        print(f"\nProcessing {parquet_file.name}...")
        df = pd.read_parquet(parquet_file)
        print(f"  Loaded {len(df)} images")

        for idx, row in df.iterrows():
            # Extract image data
            image_bytes = row['image']['bytes']
            image_filename = f"image_{image_id:06d}.jpg"
            image_path = os.path.join(image_output_folder,image_filename)

            # Save image to disk
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            # Add image info to COCO
            coco_data['images'].append({
                "id": image_id,
                "width": int(row['width']),
                "height": int(row['height']),
                "file_name": "images/" + image_filename,
                "license": 0,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            # Process annotations for this image
            objects = row['objects']
            bboxes = objects['bbox']
            categories = objects['categories']
            areas = objects['area']

            # Handle the case where bbox might be a single array or array of arrays
            if len(bboxes) > 0:

                for i in range(len(bboxes)):
                    bbox = bboxes[i]
                    # Convert to list and ensure COCO format [x, y, width, height]
                    bbox_list = [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

                    coco_data['annotations'].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": int(categories[i]),
                        "bbox": bbox_list,
                        "area": float(areas[i]),
                        "segmentation": [],  # No segmentation data in this dataset
                        "iscrowd": 0
                    })
                    annotation_id += 1

            # ...if this image has boxes

            image_id += 1

            # Progress indicator
            if idx > 0 and (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} images...")

        # ...for each row

    # ...for each parquet file

    # Write COCO JSON file
    print(f"\nWriting COCO JSON to {coco_json_path}...")
    with open(coco_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("Conversion complete!")
    print("="*60)
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")
    print(f"\nOutput directory: {output_dir}")
    print(f"  Images: {image_output_folder}")
    print(f"  Annotations: {coco_json_path}")
    print("="*60)

# ...def convert_savmap_to_coco()


#%% Interactive driver

if False:

    pass

    #%%

    convert_savmap_to_coco()


    #%% Validate output file

    from megadetector.data_management.databases.integrity_check_json_db import \
        integrity_check_json_db, IntegrityCheckOptions

    options = IntegrityCheckOptions()

    options.baseDir = coco_output_folder
    options.bCheckImageSizes = True
    options.bCheckImageExistence = True
    options.bFindUnusedImages = True
    options.bRequireLocation = False
    options.iMaxNumImages = -1
    options.nThreads = 4
    options.parallelizeWithThreads = False
    options.verbose = True
    options.allowIntIDs = True
    options.requireInfo = True
    options.validateBoxes = 'error'

    sorted_categories, data, error_info = integrity_check_json_db(
        json_file=coco_json_path,
        options=options)

    assert len(error_info['unused_files']) == 0
    assert len(error_info['validation_errors']) == 0

    """
    Found 3545 unannotated images, 195 images with multiple annotations
    Found 0 unused image files
    Found 0 unused categories

    DB contains 3924 images, 1283 annotations, 1283 bboxes, 1 categories, no sequence info

    Categories and annotation (not image) counts:

    1283 animal
    """


    #%% Visualize output file

    preview_dir = '/mnt/g/temp/savmap-coco-preview'

    from megadetector.visualization.visualize_db import \
        visualize_db, DbVizOptions

    options = DbVizOptions()

    options.num_to_visualize = 200
    options.viz_size = (1000, -1)
    # options.html_options = write_html_image_list()
    options.sort_by_filename = True
    options.trim_to_images_with_bboxes = True
    options.random_seed = 0
    options.include_filename_links = True
    options.box_thickness = 4
    options.box_expansion = 0
    options.classes_to_include = None
    options.classes_to_exclude = None
    options.parallelize_rendering = True
    options.parallelize_rendering_with_threads = True
    options.parallelize_rendering_n_cores = 16
    options.show_full_paths = False
    options.extra_image_fields_to_print = None
    options.extra_annotation_fields_to_print = None
    options.force_rendering = True
    options.verbose = False

    html_filename,_ = visualize_db(db_path=coco_json_path,
                                   output_dir=preview_dir,
                                   image_base_dir=coco_output_folder,
                                   options=options)

    from megadetector.utils.path_utils import open_file
    open_file(html_filename)


#%% Command-line driver

if __name__ == "__main__":
    convert_savmap_to_coco()
