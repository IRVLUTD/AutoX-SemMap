# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

import os
import random
import logging
import torch
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from PIL import (Image as PILImg, ImageDraw)
import cv2


def apply_matplotlib_colormap(depth_pil, colormap_name='inferno'):
    """
    Apply a matplotlib colormap to the input depth image.

    Args:
        depth_pil (PIL.Image): Input depth map image.
        colormap_name (str): Name of the matplotlib colormap to use. Default is 'inferno'.

    Returns:
        PIL.Image: Image object representing the depth map with colormap.
    """
    try:
        # Convert PIL image to numpy array
        depth_array = np.array(depth_pil)

        # Normalize depth values to range [0, 1]
        depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

        # Create colormap
        cmap = plt.get_cmap(colormap_name)

        # Apply the colormap to depth values
        colored_depth = (cmap(depth_normalized) * 255).astype(np.uint8)

        # Convert numpy array to PIL image
        colored_depth_pil = PILImg.fromarray(colored_depth)

        return colored_depth_pil

    except Exception as e:
        # Log error
        print(f"An error occurred: {e}")
        raise e


def file_exists(file_path):
    """
    Check if a file exists.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - bool: True if the file exists, False otherwise.
    """
    try:
        return os.path.exists(file_path)
    except Exception as e:
        logging.error(f"Error checking file existence: {e}")
        raise e


def crop_images(original_image, bounding_boxes):
    """
    Crop the input image using the provided bounding boxes.

    Parameters:
    - original_image (PIL.Image): Original input image.
    - bounding_boxes (list): List of bounding boxes [x_min, y_min, x_max, y_max].

    Returns:
    - cropped_images (list): List of cropped images.

    Raises:
    - ValueError: If the bounding box dimensions are invalid.
    """
    cropped_images = []

    try:
        for box in bounding_boxes:
            if len(box) != 4:
                raise ValueError("Bounding box should have 4 values: [x_min, y_min, x_max, y_max]")

            x_min, y_min, x_max, y_max = box

            # Check if the bounding box dimensions are valid
            if x_min < 0 or y_min < 0 or x_max <= x_min or y_max <= y_min:
                raise ValueError("Invalid bounding box dimensions")

            # Crop the image using the bounding box
            cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
            cropped_images.append(cropped_image)

    except ValueError as e:
        print(f"Error in crop_images: {e}")


def annotate(image_source, boxes, logits, phrases):
    """
    Annotate image with bounding boxes, logits, and phrases.

    Parameters:
    - image_source (PIL.Image): Input image source.
    - boxes (torch.tensor): Bounding boxes in xyxy format.
    - logits (list): List of confidence logits.
    - phrases (list): List of phrases.

    Returns:
    - PIL.Image: Annotated image.
    """
    try:
        # detections = sv.Detections(xyxy=boxes.cpu().numpy())
        # Deleted above line in order to create detections with dummy class_id as seen in next section
        # Create detections with dummy class_id
        detections = sv.Detections(
            xyxy=boxes.cpu().numpy(),
            class_id=np.zeros(len(boxes), dtype=int)  # Dummy class_id for all detections
        )

        labels = [
            f"{phrase} {logit:.2f}"
            for phrase, logit
            in zip(phrases, logits)
        ]
        box_annotator = sv.BoxAnnotator()

        # Annotate the image with bounding boxes
        scene = np.array(image_source)
        annotated_scene = box_annotator.annotate(scene=scene, detections=detections)

        # Draw the labels on the image
        for i, (box, label) in enumerate(zip(detections.xyxy, labels)):
            x1, y1, x2, y2 = box.astype(int)
            cv2.putText(annotated_scene, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        img_pil = PILImg.fromarray(annotated_scene)
        return img_pil
    
    except Exception as e:
        logging.error(f"Error during annotation: {e}")
        raise e


def draw_mask(mask, draw, random_color=False):
    """
    Draw a segmentation mask on an image.

    Parameters:
    - mask (numpy.ndarray): The segmentation mask as a NumPy array.
    - draw (PIL.ImageDraw.ImageDraw): The PIL ImageDraw object to draw on.
    - random_color (bool, optional): Whether to use a random color for the mask. Default is False.

    Returns:
    - None
    """
    try:
        # Define the color for the mask
        if random_color:
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 153)
        else:
            color = (30, 144, 255, 153)

        # Get the coordinates of non-zero elements in the mask
        nonzero_coords = np.transpose(np.nonzero(mask))

        # Draw each non-zero coordinate on the image
        for coord in nonzero_coords:
            draw.point(coord[::-1], fill=color)

    except Exception as e:
        logging.error(f"Error drawing mask: {e}")
        raise e


def overlay_masks(image_pil: PILImg, masks):
    """
    Overlay segmentation masks on the input image.

    Parameters:
    - image_pil (PIL.Image): The input image as a PIL image.
    - masks (List[Tensor]): List of segmentation masks as torch Tensors.

    Returns:
    - PIL.Image: The image with overlayed segmentation masks.
    """
    try:
        mask_image = PILImg.new('RGBA', image_pil.size, color=(0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_image)
        for mask in masks:
            draw_mask(mask[0].cpu().numpy(), mask_draw, random_color=True)

        image_pil = image_pil.convert('RGBA')
        image_pil.alpha_composite(mask_image)
        return image_pil.convert('RGB')

    except Exception as e:
        logging.error(f"Error overlaying masks: {e}")
        raise e


def combine_masks(gt_masks):
    """
    Combine several bit masks [N, H, W] into a mask [H,W],
    e.g. 8*480*640 tensor becomes a numpy array of 480*640.
    [[1,0,0], [0,1,0]] = > [1,2,0].

    Args:
        gt_masks (torch.Tensor): Tensor of shape [N, H, W] representing multiple bit masks.

    Returns:
        torch.Tensor: Combined mask of shape [H, W].
    """
    try:
        gt_masks = torch.flip(gt_masks, dims=(0,))
        num, h, w = gt_masks.shape
        bin_mask = torch.zeros((h, w), device=gt_masks.device)
        num_instance = len(gt_masks)

        # if there is not any instance, just return a mask full of 0s.
        if num_instance == 0:
            return bin_mask

        for m, object_label in zip(gt_masks, range(1, 1 + num_instance)):
            label_pos = torch.nonzero(m, as_tuple=True)
            bin_mask[label_pos] = object_label
        return bin_mask

    except Exception as e:
        logging.error(f"Error combining masks: {e}")
        raise e


def filter_large_boxes(boxes, w, h, threshold=0.5):
    """
    Filter out large boxes from a list of bounding boxes based on a threshold.

    Args:
        boxes (torch.Tensor): Bounding boxes of shape [N, 4].
        w (int): Width of the image.
        h (int): Height of the image.
        threshold (float, optional): Threshold value for filtering large boxes. Defaults to 0.5.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Filtered bounding boxes and corresponding indices.
    """
    try:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1) * (y2 - y1)
        index = area < (w * h) * threshold
        return boxes[index], index.cpu()

    except Exception as e:
        logging.error(f"Error filtering large boxes: {e}")
        raise e


def save_mask(masks, output_path, image_path, phrases, conf):
    """
    Saves masks from the SAM predict function and then saves 
    them into seperate folders for each class of each 
    image sent through the pipeline.

    Parameters: 
    - masks (torch.Tensor): Tensor of shape [N, H, W] holding all masks 
        of the given image.
    - output_path (str): Path to the output directory where a folder called (segments) will be
        created and masks will be saved.
    - image_path (str): Path to the input image (image that the masks were created from).
    - phrases (list): List of phrases corresponding to each mask.
    
    Returns:
    - None (creates a folder in the output_path which will contain subfolders
            corresponding to the image number and subfolders corresponding 
            to the detected class)
    """


    try:
        mask_arrays = masks.cpu().numpy()
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        if image_name.endswith('_color'):
            image_name = image_name[:-6]

        image_output_path = os.path.join(output_path, image_name)
        os.makedirs(image_output_path, exist_ok=True)

        prompt_mask_counts = {}

        for i, mask in enumerate(mask_arrays):
            mask = mask[0]

            mask = mask.astype(np.float32)
            mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255.0
            mask = mask.astype(np.uint8)

            img = PILImg.fromarray(mask)
            phrase = phrases[i]
            conf_ =  conf[i]

            if phrase not in prompt_mask_counts:
                prompt_mask_counts[phrase] = 0

            prompt_output_path = os.path.join(image_output_path, phrase)
            os.makedirs(prompt_output_path, exist_ok=True)

            mask_index = prompt_mask_counts[phrase]
            mask_filename = f"mask_{phrase}_{mask_index}_{conf_}.png"
            mask_image_path = os.path.join(prompt_output_path, mask_filename)

            img.save(mask_image_path)
            prompt_mask_counts[phrase] += 1

    except Exception as e:
        logging.error(f"Error saving masks: {e}")
        raise e
    

def filter(bboxes, conf_list, phrases ,conf_bound, yVal, precentWidth=0.5, precentHeight=0.5, precentArea=0.05, filterChoice=True):

    """
    Filters out false positives from detections:
    - Noise from floor 
    - Detections of the entire image (Very large, false detections)
    - False detections of doors (light glare, windows, etc.)

    Parameters:
    - bboxes (torch.Tensor): Tensor of bounding boxes with shape [N, 4], where each row is structured by
                             (center x, center y, width (% of screen), height (% of screen)).
    - conf_list (torch.Tensor): Tensor of confidence scores corresponding to each bounding box.
    - phrases (list): List of phrases corresponding to each bounding box.
    - conf_bound (float): Confidence score upper bound for filtering.
        range: 0.0-1.0
    - yVal (float): Y-coordinate lower bound for filtering (if the bboxes center y coordinate is lower than yVal it gets removed) (% of screen) 
        range: 0.0-1.0
        0 is top 1 is bottom
    - precentWidth (float, optional): Maximum width of bounding boxes as a percentage of image width. Default is 0.5 (50%).
        range: 0.0-1.0
    - precentHeight (float, optional): Maximum height of bounding boxes as a percentage of image height. Default is 0.5 (50%).
        range: 0.0-1.0
    - precentArea (float, optional): Minimum area of bounding boxes as a percentage of total image area. Default is 0.05 (5%).
        range: 0.0-1.0
    - filterChoice (bool, optional): Whether to apply filtering logic. Default is True so the filter runs by default. By changing to False the filter
        not apply to the detections.

    Returns:
    - bboxes (torch.Tensor): Filtered bounding boxes
    - conf_list (torch.Tensor): Filtered confidence scores
    - phrases (list): Filtered list of phrases
    - flag (boolean): indicating if there are any detections left after filter
            false = no detections left (breaks loop)
            true =  some detections left (continuted) 

    """

    if filterChoice:
        IMAGE_WIDTH = 640
        IMAGE_HEIGHT = 480
        IMAGE_AREA = IMAGE_WIDTH * IMAGE_HEIGHT
        MIN_BOX_AREA = precentArea * IMAGE_AREA 

        phrases_np = np.array(phrases)

        if conf_list.size(dim=0) >= 1:
            c1 = bboxes[:, 3] <= precentHeight  # taller than 50% (Default)
            c2 = bboxes[:, 2] <= precentWidth  # wider than 50% (Default)
            c3 = bboxes[:, 1] <= yVal
            
            # Calculate area of each bounding box and filter out those with area < 5% of total image (Default)
            box_areas = (bboxes[:, 2] * IMAGE_WIDTH) * (bboxes[:, 3] * IMAGE_HEIGHT)
            c4 = box_areas >= MIN_BOX_AREA

            mask = c1 & c2 & c3 & c4

            # specific filter for doors
            door_indices = np.where(phrases_np == 'door')[0]
            for i in door_indices:
                width = bboxes[i, 2] * IMAGE_WIDTH
                height = bboxes[i, 3] * IMAGE_HEIGHT
                if height / width < 1.7 or box_areas[i] < 0.04 * IMAGE_AREA:
                    mask[i] = False

            bboxes = bboxes[mask]
            conf_list = conf_list[mask]

            phrases_np = phrases_np[mask.cpu().numpy()]

        # Filters out images with detections at all
        if conf_list.size(dim=0) == 0:
            return bboxes, conf_list, phrases_np.tolist(), True

        # Creates an upper bound for confidence
        if any(conf >= conf_bound for conf in conf_list):
            return bboxes, conf_list, phrases_np.tolist(), True 

        return bboxes, conf_list, phrases_np.tolist(), False

    # If filterChoice is False, skip the filtering. Flag is returned as False if there are detections
    if not filterChoice:
        if conf_list.size(dim=0) == 0:
            return bboxes, conf_list, phrases, True
        return bboxes, conf_list, phrases, False
