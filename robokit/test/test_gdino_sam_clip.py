# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

import numpy as np
from absl import app, logging
from PIL import Image as PILImg
from robokit.utils import annotate, overlay_masks
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor, ZeroShotClipPredictor


def main(argv):
    # Path to the input image
    image_path = argv[0]
    text_prompt =  'objects'

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()
        SAM = SegmentAnythingPredictor()
        _clip = ZeroShotClipPredictor()

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")
        
        logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
        bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

        logging.info("GDINO post processing")
        # Get image width and height
        w, h = image_pil.size
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
        logging.info("SAM prediction done")

        overlay_masks(image_pil,masks)

        logging.info("Crop images based on bounding boxes")
        cropped_bbox_imgs = list(map(lambda bbox: (image_pil.crop(bbox.int().numpy())), image_pil_bboxes))
        
        logging.info("CLIP: Predict labels for cropped images")
        clip_conf, idx = _clip.predict(cropped_bbox_imgs, text_prompt.split(','))
        
        logging.info("Scale the original image for visualization")
        scaled_image_pil = gdino.image_transform_for_vis(image_pil)
        
        # Get the size of the scaled image
        ws, hs = scaled_image_pil.size
        
        logging.info("Scale bounding boxes to match the scaled image size")
        scaled_image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, ws, hs)

        logging.info("Get the predicted labels based on the indices")
        predicted_labels = [ text_prompt.split(',')[i] for i in idx ]
        
        logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        # bbox_annotated_pil = annotate(scaled_image_pil, scaled_image_pil_bboxes, clip_conf, predicted_labels)
        bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, clip_conf, predicted_labels)

        bbox_annotated_pil.show()

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    app.run(main, ['imgs/irvl-clutter-test.png'])
