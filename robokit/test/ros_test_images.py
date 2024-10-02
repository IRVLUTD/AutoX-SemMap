#!/usr/bin/env python

"""Test GroundingSAM on ros images"""

import rospy
import threading
import ros_numpy
import numpy as np
import message_filters
from PIL import Image as PILImg
from sensor_msgs.msg import Image, CameraInfo
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
lock = threading.Lock()


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


class ImageListener:

    def __init__(self, camera='Fetch'):

        self.im = None
        self.depth = None
        self.rgb_frame_id = None
        self.rgb_frame_stamp = None
        self.counter = 0
        self.output_dir = 'output/real_world'

        # initialize network
        self.text_prompt =  'objects'          
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()     

        # initialize a node
        rospy.init_node("seg_rgb")
        self.label_pub = rospy.Publisher('seg_label_refined', Image, queue_size=10)
        self.score_pub = rospy.Publisher('seg_score', Image, queue_size=10)     
        self.image_pub = rospy.Publisher('seg_image', Image, queue_size=10)

        if camera  == 'Fetch':
            self.base_frame = 'base_link'
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
            self.camera_frame = 'head_camera_rgb_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Realsense':
            # use RealSense D435
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
            self.camera_frame = 'measured/camera_color_optical_frame'
            self.target_frame = self.base_frame
        elif camera == 'Azure':
            self.base_frame = 'measured/base_link'
            rgb_sub = message_filters.Subscriber('/k4a/rgb/image_raw', Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/k4a/depth_to_rgb/image_raw', Image, queue_size=10)
            msg = rospy.wait_for_message('/k4a/rgb/camera_info', CameraInfo)
            self.camera_frame = 'rgb_camera_link'
            self.target_frame = self.base_frame
        else:
            # use kinect
            self.base_frame = '%s_rgb_optical_frame' % (camera)
            rgb_sub = message_filters.Subscriber('/%s/rgb/image_color' % (camera), Image, queue_size=10)
            depth_sub = message_filters.Subscriber('/%s/depth_registered/image' % (camera), Image, queue_size=10)
            msg = rospy.wait_for_message('/%s/rgb/camera_info' % (camera), CameraInfo)
            self.camera_frame = '%s_rgb_optical_frame' % (camera)
            self.target_frame = self.base_frame

        # update camera intrinsics
        intrinsics = np.array(msg.K).reshape(3, 3)
        self.fx = intrinsics[0, 0]
        self.fy = intrinsics[1, 1]
        self.px = intrinsics[0, 2]
        self.py = intrinsics[1, 2]
        print(intrinsics)

        queue_size = 1
        slop_seconds = 0.1
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback_rgbd)


    def callback_rgbd(self, rgb, depth):

        if depth.encoding == '32FC1':
            depth_cv = ros_numpy.numpify(depth)
        elif depth.encoding == '16UC1':
            depth_cv = ros_numpy.numpify(depth).copy().astype(np.float32)
            depth_cv /= 1000.0
        else:
            rospy.logerr_throttle(
                1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(
                    depth.encoding))
            return

        im = ros_numpy.numpify(rgb)

        with lock:
            self.im = im.copy()
            self.depth = depth_cv.copy()
            self.rgb_frame_id = rgb.header.frame_id
            self.rgb_frame_stamp = rgb.header.stamp


    def run_network(self):

        with lock:
            if listener.im is None:
              return
            im_color = self.im.copy()
            depth_img = self.depth.copy()
            rgb_frame_id = self.rgb_frame_id
            rgb_frame_stamp = self.rgb_frame_stamp

        print('===========================================')

        # bgr image
        im = im_color.astype(np.uint8)[:, :, (2, 1, 0)]
        img_pil = PILImg.fromarray(im)
        bboxes, phrases, gdino_conf = self.gdino.predict(img_pil, self.text_prompt)

        # Scale bounding boxes to match the original image size
        w = im.shape[1]
        h = im.shape[0]
        image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        # logging.info("SAM prediction")
        image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)

        # filter large boxes
        image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
        masks = masks[index]
        mask = combine_masks(masks[:, 0, :, :]).cpu().numpy()
        gdino_conf = gdino_conf[index]
        ind = np.where(index)[0]
        phrases = [phrases[i] for i in ind]

        # logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases)
        # bbox_annotated_pil.show()
        im_label = np.array(bbox_annotated_pil)

        # show result
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 2, 1)
        # plt.imshow(im_label)
        # ax.set_title('output image')
        # ax = fig.add_subplot(1, 2, 2)
        # plt.imshow(mask)
        # ax.set_title('mask')              
        # plt.show()        

        # publish segmentation mask
        label = mask
        label_msg = ros_numpy.msgify(Image, label.astype(np.uint8), 'mono8')
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.label_pub.publish(label_msg)

        # publish score map
        score = label.copy()
        mask_ids = np.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]
        for index, mask_id in enumerate(mask_ids):
            score[label == mask_id] = gdino_conf[index]
        label_msg = ros_numpy.msgify(Image, score.astype(np.uint8), 'mono8')
        label_msg.header.stamp = rgb_frame_stamp
        label_msg.header.frame_id = rgb_frame_id
        label_msg.encoding = 'mono8'
        self.score_pub.publish(label_msg)        

        num_object = len(np.unique(label)) - 1
        print('%d objects' % (num_object))

        # publish segmentation images
        rgb_msg = ros_numpy.msgify(Image, im_label, 'rgb8')
        rgb_msg.header.stamp = rgb_frame_stamp
        rgb_msg.header.frame_id = rgb_frame_id
        self.image_pub.publish(rgb_msg)


if __name__ == '__main__':
    # image listener
    listener = ImageListener()
    while not rospy.is_shutdown():
       listener.run_network()
