# (c) 2024 Jishnu Jaykumar Padalunkal.
# Work done while being at the Intelligent Robotics and Vision Lab at the University of Texas, Dallas
# Please check the licenses of the respective works utilized here before using this script.


import os
import clip
import torch
import logging
import warnings
import numpy as np
from PIL import Image as PILImg
import torchvision.transforms as tvT
from featup.util import norm, unnorm
from featup.plotting import plot_feats
from torchvision.ops import box_convert
from huggingface_hub import hf_hub_download
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.inference import predict
from groundingdino.util.utils import clean_state_dict
# from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


os.system("python setup.py build develop --user")
os.system("pip install packaging==21.3")
warnings.filterwarnings("ignore")


class Logger(object):
    """
    This is a logger class.

    Attributes:
        logger: Logger instance for logging.
    """
    def __init__(self):
        """
        Initializes the Logger class.
        """
        super(Logger, self).__init__()

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


class Device(object):
    """
    This is a device class.

    Attributes:
        device (str): The device type ('cuda' or 'cpu').
        logger: Logger instance for logging.
    """
    def __init__(self):
        """
        Initializes the Device class.
        """
        super(Device, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


class CommonContextObject(Logger, Device):
    """
    This is a common context object class.

    Attributes:
        logger: Logger instance for logging.
        device (str): The device type ('cuda' or 'cpu').
    """
    def __init__(self):
        """
        Initializes the CommonContextObject class.
        """
        super(CommonContextObject, self).__init__()


class FeatureUpSampler(CommonContextObject):
    """
    Root class for feature upsampling.
    All other feature upsampling classes should inherit this.

    Attributes:
        logger: Logger instance for logging errors.
    """
    def __init__(self):
        """
        Initializes the DepthPredictor class.
        """
        super(FeatureUpSampler, self).__init__()

    def upsample(self):
        """
        Upsample method for feature upscaling.
        Raises NotImplementedError as it should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        try:
            raise NotImplementedError("Upsample method must be implemented by subclasses")
        except NotImplementedError as e:
            self.logger.error(f"Error in upsample method: {e}")
            raise e


class DepthPredictor(CommonContextObject):
    """
    Root class for depth prediction.
    All other depth prediction classes should inherit this.

    Attributes:
        logger: Logger instance for logging errors.
    """
    def __init__(self):
        """
        Initializes the DepthPredictor class.
        """
        super(DepthPredictor, self).__init__()

    def predict(self):
        """
        Predict method for depth prediction.
        Raises NotImplementedError as it should be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented by subclasses.
        """
        try:
            raise NotImplementedError("predict method must be implemented by subclasses")
        except NotImplementedError as e:
            self.logger.error(f"Error in predict method: {e}")
            raise e


class FeatUp(FeatureUpSampler):
    """
    A class for upsampling features using a pre-trained backbone model.

    Attributes:
        input_size (int): Input size of the images.
        backbone_alias (str): Alias of the pre-trained backbone model.
        upsampler (torch.nn.Module): Feature upsampling module.
        logger (logging.Logger): Logger object for logging.
    """

    def __init__(self, backbone_alias, input_size, visualize_output=False):
        """
        Initializes the FeatUp class.

        Args:
            backbone_alias (str): Alias of the pre-trained backbone model.
            input_size (int): Input size of the images.
        """
        super(FeatUp, self).__init__()
        self.input_size = input_size
        self.backbone_alias = backbone_alias
        self.visualize_output = visualize_output
        self.img_transform = tvT.Compose([
            tvT.Resize(self.input_size),
            tvT.CenterCrop((self.input_size, self.input_size)),
            tvT.ToTensor(),
            norm
        ])
        try:
            self.upsampler = torch.hub.load("mhamilton723/FeatUp", self.backbone_alias).to(self.device)
        except Exception as e:
            self.logger.error(f"Error loading FeatUp model: {e}")
            raise e

    def upsample(self, image_tensor):
        """
        Upsamples the features of encoded input image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            Tuple: A tuple containing the original image tensor, backbone features, and upsampled features.
        """
        try:
            image_tensor = image_tensor.to(self.device)
            upsampled_features = self.upsampler(image_tensor) # upsampled features using backbone features; high resolution
            backbone_features = self.upsampler.model(image_tensor) # backbone features; low resolution
            orig_image = unnorm(image_tensor)
            batch_size = orig_image.shape[0]
            if self.visualize_output:
                self.logger.info("Plot input image with backbone and upsampled output")
                for i in range(batch_size):
                    plot_feats(orig_image[i], backbone_features[i], upsampled_features[i])
            return orig_image, backbone_features, upsampled_features

        except Exception as e:
            self.logger.error(f"Error during feature upsampling: {e}")
            raise e


class DepthAnythingPredictor(DepthPredictor):
    """
    A predictor class for depth estimation using a pre-trained model.

    Attributes:
        image_processor: Pre-trained image processor.
        model: Pre-trained depth estimation model.
        logger: Logger instance for logging errors.
    """
    def __init__(self):
        """
        Initializes the DepthAnythingPredictor class.
        """
        super(DepthPredictor, self).__init__() 
        self.image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")
        self.logger = logging.getLogger(__name__)

    def predict(self, img_pil):
        """
        Predicts depth from an input image.

        Args:
            PIL Image: Input image.

        Returns:
            PIL Image: Predicted depth map as a PIL image.
            numpy.ndarray: Predicted depth values as a numpy array.
        """
        try:
            image = img_pil.convert('RGB')

            # prepare image for the model
            inputs = self.image_processor(images=image, return_tensors="pt")

            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth

            # interpolate to original size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            )

            # visualize the prediction
            output = prediction.squeeze().cpu().numpy()
            formatted = (output * 255 / np.max(output)).astype("uint8")
            depth_pil = PILImg.fromarray(formatted)
            return depth_pil, output

        except Exception as e:
            self.logger.error(f"Error predicting depth: {e}")
            raise e


class ObjectPredictor(CommonContextObject):
    """
    Root class for object predicton
    All other object prediction classes should inherit this
    """
    def __init__(self):
        super(ObjectPredictor, self).__init__()
    

    def bbox_to_scaled_xyxy(self, bboxes: torch.tensor, img_w, img_h):
        """
        Convert bounding boxes to scaled xyxy format.

        Parameters:
        - bboxes (torch.tensor): Input bounding boxes in cxcywh format.
        - img_w (int): Image width.
        - img_h (int): Image height.

        Returns:
        - torch.tensor: Converted bounding boxes in xyxy format.
        """
        try:
            bboxes = bboxes * torch.Tensor([img_w, img_h, img_w, img_h])
            bboxes_xyxy = box_convert(boxes=bboxes, in_fmt="cxcywh", out_fmt="xyxy")
            return bboxes_xyxy
        
        except Exception as e:
            self.logger.error(f"Error during bounding box conversion: {e}")
            raise e


class GroundingDINOObjectPredictor(ObjectPredictor):
    """
    This class implements Object detection using HuggingFace GroundingDINO
    Here instead of using generic language query, we fix the text prompt as "objects" which enables
    getting compact bounding boxes arounds generic objects.
    Hope is that these cropped bboxes when used with OpenAI CLIP yields good classification results.
    """
    def __init__(self):
        super(GroundingDINOObjectPredictor, self).__init__()
        self.ckpt_repo_id = "ShilongLiu/GroundingDINO"
        self.ckpt_filenmae = "groundingdino_swint_ogc.pth"
        self.config_file = "robokit/cfg/gdino/GroundingDINO_SwinT_OGC.py"
        self.model = self.load_model_hf(
            self.config_file, self.ckpt_repo_id, self.ckpt_filenmae
        )
    

    def load_model_hf(self, model_config_path, repo_id, filename):
        """
        Load model from Hugging Face hub.

        Parameters:
        - model_config_path (str): Path to model configuration file.
        - repo_id (str): ID of the repository.
        - filename (str): Name of the file.

        Returns:
        - torch.nn.Module: Loaded model.
        """
        try:
            args = SLConfig.fromfile(model_config_path) 
            model = build_model(args)
            args.device = self.device

            cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(cache_file, map_location=self.device)
            log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
            print("Model loaded from {} \n => {}".format(cache_file, log))
            _ = model.eval()
            return model    

        except Exception as e:
            # Log error and raise exception
            self.logger.error(f"Error loading model from Hugging Face hub: {e}")
            raise e

    def image_transform_grounding(self, image_pil):
        """
        Apply image transformation for grounding.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - tuple: Tuple containing original PIL image and transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image_pil, image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for grounding: {e}")
            raise e

    def image_transform_for_vis(self, image_pil):
        """
        Apply image transformation for visualization.

        Parameters:
        - image_pil (PIL.Image): Input image.

        Returns:
        - torch.tensor: Transformed tensor image.
        """
        try:
            transform = T.Compose([
                T.RandomResize([800], max_size=1333),
            ])
            image, _ = transform(image_pil, None) # 3, h, w
            return image
        
        except Exception as e:
            self.logger.error(f"Error during image transformation for visualization: {e}")
            raise e
    
    # Edited to add box_threshold and text_threshold as arguments with default value of 0.35 (Itay Kadosh)
    def predict(self, image_pil: PILImg, det_text_prompt: str = "objects", box_threshold = 0.35, text_threshold=0.35):
        """
        Get predictions for a given image using GroundingDINO model.
        Paper: https://arxiv.org/abs/2303.05499
        Parameters:
        - image_pil (PIL.Image): PIL.Image representing the input image.
        - det_text_prompt (str): Text prompt for object detection
        Returns:
        - bboxes (list): List of normalized bounding boxeS in cxcywh
        - phrases (list): List of detected phrases.
        - conf (list): List of confidences.

        Raises:
        - Exception: If an error occurs during model prediction.
        """
        try:
            _, image_tensor = self.image_transform_grounding(image_pil)
            bboxes, conf, phrases = predict(self.model, image_tensor, det_text_prompt, box_threshold=box_threshold, text_threshold=text_threshold, device=self.device)
            return bboxes, phrases, conf        
        except Exception as e:
            self.logger.error(f"Error during model prediction: {e}")
            raise e


class SegmentAnythingPredictor(ObjectPredictor):
    """
    Predictor class for segmenting objects using the Segment Anything model.

    Inherits from ObjectPredictor.

    Attributes:
    - device (str): The device used for inference, either "cuda" or "cpu".
    - sam (torch.nn.Module): The Segment Anything model.
    - mask_generator (SamAutomaticMaskGenerator): The mask generator for the SAM model.
    - predictor (SamPredictor): The predictor for the SAM model.
    """

    def __init__(self):
        """
        Initialize the SegmentAnythingPredictor object.
        """
        super(SegmentAnythingPredictor, self).__init__()
        self.sam = sam_model_registry["vit_t"](checkpoint="ckpts/mobilesam/vit_t.pth")
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)  # generate masks for entire image
        self.sam.to(device=self.device)
        self.sam.eval()
        self.predictor = SamPredictor(self.sam)

    def predict(self, image, prompt_bboxes):
        """
        Predict segmentation masks for the input image.

        Parameters:
        - image: The input image as a numpy array.
        - prompt_bboxes: Optional prompt bounding boxes as a list of lists of integers [x_min, y_min, x_max, y_max].

        Returns:
        - A tuple containing the input bounding boxes (if provided) and the segmentation masks as torch Tensors.

        Raises:
        - ValueError: If the input image is not a numpy array.
        """
        try:
            # Convert input image to numpy array
            image = np.array(image)

            # Check if prompt_bboxes is provided
            if prompt_bboxes is not None:
                # Convert prompt bounding boxes to torch tensor
                input_boxes = torch.tensor(prompt_bboxes, device=self.predictor.device)
                transformed_boxes = self.predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
                self.predictor.set_image(image)
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
            else:
                input_boxes = None
                masks = self.mask_generator.generate(image)
            
            return input_boxes, masks

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return None, None


class ZeroShotClipPredictor(CommonContextObject):
    def __init__(self):
        super(ZeroShotClipPredictor, self).__init__()
        
        # Load the CLIP model
        self.model, self.preprocess = clip.load('ViT-L/14@336px', self.device)
        self.model.eval()

    def get_features(self, images, text_prompts):
        """
        Extract features from a list of images and text prompts.

        Parameters:
        - images (list of PIL.Image): A list of PIL.Image representing images.
        - text_prompts (list of str): List of text prompts.

        Returns:
        - Tuple of numpy.ndarray: Concatenated image features and text features as numpy arrays.

        Raises:
        - ValueError: If images is not a tensor or a list of tensors.
        - RuntimeError: If an error occurs during feature extraction.
        """
        try:

            with torch.no_grad():
                text_inputs = torch.cat([clip.tokenize(prompt) for prompt in text_prompts]).to(self.device)
                _images = torch.stack([self.preprocess(img) for img in images]).to(self.device)
                img_features = self.model.encode_image(_images)
                text_features = self.model.encode_text(text_inputs)
            
            return img_features, text_features

        except ValueError as ve:
            self.logger.error(f"ValueError in get_image_features: {ve}")
            raise ve
        except RuntimeError as re:
            self.logger.error(f"RuntimeError in get_image_features: {re}")
            raise re

    def predict(self, image_array, text_prompts):
        """
        Run zero-shot prediction using CLIP model.

        Parameters:
        - image_array (List[torch.tensor]): List of tensor images.
        - text_prompts (list): List of text prompts for prediction.

        Returns:
        - Tuple: Tuple containing prediction confidence and indices.
        """
        try:
            # Perform prediction
            image_features, text_features = self.get_features(image_array, text_prompts)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            pconf, indices = similarity.topk(1)

            return (pconf.flatten(), indices.flatten())

        except Exception as e:
            # Log error and raise exception
            self.logger.error(f"Error during prediction: {e}")
            raise e
