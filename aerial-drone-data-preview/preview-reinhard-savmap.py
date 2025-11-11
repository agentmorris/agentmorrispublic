"""
Code to render sample images and points in the Zenodo version of the SAVMAP dataset:

https://zenodo.org/records/1204408

Annotations are polygons in a .geojson file.

We're actually going to render boxes, not polygons, and we're going to collapse overlapping
annotations into larger boxes.  The raw polygons are approximate.

"""

#%% Imports and constants

import os

base_folder = r'G:\temp\savmap_dataset_v2'
annotations_file = os.path.join(base_folder,'savmap_annotations_2014.geojson')
assert os.path.isfile(annotations_file)
output_folder = r'g:\temp\savmap_preview'
os.makedirs(output_folder,exist_ok=True)


#%% Render boxes

import json
import os
import random
from PIL import Image, ImageDraw
import collections
import numpy as np

random.seed(0)

# Assuming these variables are defined already
# annotations_file = "path/to/savmap_annotations_2014_sampled.geojson"
# base_folder = "path/to/images/"
# output_folder = "path/to/output/"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load GeoJSON file
with open(annotations_file, 'r') as f:
    geojson_data = json.load(f)

# Group annotations by image UUID
image_annotations = collections.defaultdict(list)
for feature in geojson_data['features']:
    image_uuid = feature['properties']['IMAGEUUID']
    polygon_coords = feature['geometry']['coordinates'][0]  # Get the polygon coordinates
    image_annotations[image_uuid].append(polygon_coords)

# Find images with at least min_annotations annotations
min_annotations = 5
n_images = 100
images_with_multiple_annotations = {img_id: annotations
                                   for img_id, annotations in image_annotations.items()
                                   if len(annotations) >= min_annotations}

num_samples = min(n_images, len(images_with_multiple_annotations))
sampled_image_ids = random.sample(list(images_with_multiple_annotations.keys()), num_samples)

def polygon_to_bbox(polygon_coords):
    """Convert polygon coordinates to axis-aligned bounding box [xmin, ymin, xmax, ymax]"""
    x_coords = [point[0] for point in polygon_coords]
    y_coords = [point[1] for point in polygon_coords]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def merge_overlapping_boxes(boxes):
    """Merge overlapping bounding boxes"""
    if not boxes:
        return []

    # Sort boxes by the x-coordinate of top-left corner
    sorted_boxes = sorted(boxes, key=lambda box: box[0])
    merged_boxes = [sorted_boxes[0]]

    for current_box in sorted_boxes[1:]:
        previous_box = merged_boxes[-1]

        # Check if current box overlaps with the previous box
        if (current_box[0] <= previous_box[2] and  # x_min_curr <= x_max_prev
            current_box[1] <= previous_box[3] and  # y_min_curr <= y_max_prev
            current_box[2] >= previous_box[0] and  # x_max_curr >= x_min_prev
            current_box[3] >= previous_box[1]):    # y_max_curr >= y_min_prev

            # Merge the boxes
            merged_boxes[-1] = [
                min(previous_box[0], current_box[0]),  # x_min
                min(previous_box[1], current_box[1]),  # y_min
                max(previous_box[2], current_box[2]),  # x_max
                max(previous_box[3], current_box[3])   # y_max
            ]
        else:
            # No overlap, add the current box to the result
            merged_boxes.append(current_box)

    return merged_boxes

# Process each sampled image
for image_id in sampled_image_ids:
    # Construct the image path
    image_path = os.path.join(base_folder, f"{image_id}.JPG")

    try:
        # Open the image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)

        # Convert polygons to bounding boxes
        bboxes = []
        for polygon_coords in image_annotations[image_id]:
            bbox = polygon_to_bbox(polygon_coords)
            bboxes.append(bbox)

        # Merge overlapping boxes
        merged_boxes = merge_overlapping_boxes(bboxes)

        # Draw the merged bounding boxes
        for box in merged_boxes:
            # box format: [xmin, ymin, xmax, ymax]
            draw.rectangle(
                [(box[0], box[1]), (box[2], box[3])],
                outline=(255, 0, 0),
                width=8
            )

        # Save the image with annotations
        output_path = os.path.join(output_folder, f"{image_id}.JPG")
        img.save(output_path)

        print(f"Processed image: {image_id}")
    except Exception as e:
        print(f"Error processing image {image_id}: {e}")

print(f"Saved {num_samples} annotated images to {output_folder}")
