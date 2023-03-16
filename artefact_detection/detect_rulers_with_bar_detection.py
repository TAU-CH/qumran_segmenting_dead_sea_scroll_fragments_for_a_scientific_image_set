import os
import json
import numpy as np
import cv2

# Function to create a binary mask for a given image and list of bounding boxes
def create_mask(img_shape, bboxes):
    # Create a black image with the same shape as the original image
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    # Draw a white rectangle for each bounding box on the mask image
    for bbox in bboxes:
        x, y, w, h = bbox
        mask[int(y):int(h), int(x):int(w)] = 255
    
    return mask

# Define the folder paths and file names
data_folder = 'test/images'   # folder containing the original images
ruler_positions = 'ruler_positions'   # folder to save the mask images
annotations_file = 'test_predictions.json'   # file containing the annotations

# Load the annotations file
with open(os.path.join(annotations_file)) as f:
    annotations = json.load(f)

# Create a dictionary that maps image names to bounding boxes
img_to_coords = {}
for annotation in annotations['annotations']:
    img_id = annotation['image_id']
    for img in annotations['images']:
        if img_id == img['id']:
            img_name = img['file_name']
            break
    if img_name not in img_to_coords:
        img_to_coords[img_name] = []
    img_to_coords[img_name].append(annotation['bbox'])

# Create masks for each image in the data folder
for subfolder_name in os.listdir(data_folder):
    subfolder_path = os.path.join(data_folder, subfolder_name)
    print(subfolder_name)
    if not os.path.isdir(subfolder_path):
        continue

    recto_color_img_name = f"{subfolder_name}_recto_color.jpg"
    verso_color_img_name = f"{subfolder_name}_verso_color.jpg"

    # Create mask for recto color image if there are annotations for it
    if recto_color_img_name in img_to_coords:
        recto_color_img_path = os.path.join(subfolder_path, 'recto_color.jpg')
        img = cv2.imread(recto_color_img_path)
        mask = create_mask(img.shape[:2], img_to_coords[recto_color_img_name])
        cv2.imwrite(os.path.join(ruler_positions, recto_color_img_name), mask)

    # Create mask for verso color image if there are annotations for it
    if verso_color_img_name in img_to_coords:
        verso_color_img_path = os.path.join(subfolder_path, 'verso_color.jpg')
        img = cv2.imread(verso_color_img_path)
        mask = create_mask(img.shape[:2], img_to_coords[verso_color_img_name])
        cv2.imwrite(os.path.join(ruler_positions, verso_color_img_name), mask)
