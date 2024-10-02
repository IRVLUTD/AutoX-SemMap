# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

import numpy as np
from absl import app, logging
from PIL import Image as PILImg
from robokit.perception import FeatUp


def main(argv):
    # Path to the input image
    image_path = argv[0]

    try:
        logging.info("Initialize object detectors")
        # backbone_alias options: {'resnet', 'vit', 'clip', 'dino16', 'dinov2'}
        # the input image will be resized to input size
        pix_encoder = FeatUp(backbone_alias="dinov2", input_size=252, visualize_output=True)

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")

        logging.info("Convert PIL Image to torch Tensor")
        image_tensor = pix_encoder.img_transform(image_pil).unsqueeze(0).cuda()

        logging.info("FeatUp Upsampling")
        original_image_tensor, backbone_features, upsampled_features = pix_encoder.upsample(image_tensor)

        logging.info(f"original_image_tensor.shape:{original_image_tensor.shape}, backbone_features.shape:{backbone_features.shape}, upsampled_features.shape: {upsampled_features.shape}")

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    app.run(main, ['imgs/irvl-clutter-test.png'])
