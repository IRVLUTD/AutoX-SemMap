# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.

from absl import (app, logging)
from PIL import Image as PILImg
from robokit.utils import apply_matplotlib_colormap
from robokit.perception import DepthAnythingPredictor


def main(argv):
    # Path to the input image
    image_path = argv[0]

    try:
        logging.info("Initialize object detectors")
        depth_any = DepthAnythingPredictor()

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")
        w, h =image_pil.size

        logging.info("Depth Anything prediction")
        depth_pil, raw_depth_output = depth_any.predict(image_pil)

        logging.info("Convert depth values to heatmap format")
        # colormap ref: https://github.com/yuki-koyama/pycolormap?tab=readme-ov-file
        depth_to_colomap_pil = apply_matplotlib_colormap(depth_pil, colormap_name='inferno')

        depth_to_colomap_pil.show()

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    app.run(main, ['imgs/irvl-clutter-test.png'])
