#!/usr/bin/env python3

"""Test a Gounding SAM network on an image database."""

import time
import scipy
import torch
import os, sys
import argparse
import numpy as np
from PIL import Image as PILImg
from matplotlib import pyplot as plt

from robokit.datasets.factory import get_dataset
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, combine_masks, filter_large_boxes
from robokit.evaluation import multilabel_metrics


# filter labels on zero depths
def filter_labels_depth(labels, depth, threshold):
    labels_new = labels.clone()
    for i in range(labels.shape[0]):
        label = labels[i]
        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            roi_depth = depth[i, 2][label == mask_id]
            depth_percentage = torch.sum(roi_depth > 0).float() / torch.sum(mask)
            if depth_percentage < threshold:
                labels_new[i][label == mask_id] = 0

    return labels_new


# test a dataset
def test_segnet(test_loader, gdino, SAM, output_dir, vis=False):

    text_prompt =  'objects'
    epoch_size = len(test_loader)

    metrics_all = []
    metrics_all_refined = []
    for i, sample in enumerate(test_loader):

        end = time.time()

        # construct input
        im_tensor = sample['image_color']
        im = im_tensor[0].numpy().transpose((1, 2, 0))
        depth = None
        label = sample['label'].cuda()

        # run network
        img_pil = PILImg.fromarray(im)
        bboxes, phrases, gdino_conf = gdino.predict(img_pil, text_prompt)

        # Scale bounding boxes to match the original image size
        w = im.shape[1]
        h = im.shape[0]
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        # logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(img_pil, image_pil_bboxes)

        # filter large boxes
        image_pil_bboxes, index = filter_large_boxes(image_pil_bboxes, w, h, threshold=0.5)
        masks = masks[index]
        out_label = combine_masks(masks[:, 0, :, :])

        if 'ocid' in test_loader.dataset.name and depth is not None:
            # filter labels on zero depth
            out_label = filter_labels_depth(out_label, depth, 0.5)

        if 'osd' in test_loader.dataset.name and depth is not None:
            # filter labels on zero depth
            out_label = filter_labels_depth(out_label, depth, 0.8)

        # evaluation
        gt = sample['label'].squeeze().numpy()
        prediction = out_label.squeeze().detach().cpu().numpy()
        metrics = multilabel_metrics(prediction, gt)
        metrics_all.append(metrics)
        print(metrics)

        if vis:
            gdino_conf = gdino_conf[index]
            ind = np.where(index)[0]
            phrases = [phrases[i] for i in ind]            
            bbox_annotated_pil = annotate(overlay_masks(img_pil, masks), image_pil_bboxes, gdino_conf, phrases)
            im_label = np.array(bbox_annotated_pil)
            fig = plt.figure()
            ax = fig.add_subplot(1, 3, 1)
            plt.imshow(im[:, :, (2, 1, 0)])
            ax.set_title('input image')
            ax = fig.add_subplot(1, 3, 2)
            plt.imshow(im_label)
            ax.set_title('input image')
            ax = fig.add_subplot(1, 3, 3)
            plt.imshow(prediction)
            ax.set_title('mask')              
            plt.show()
        else:
            # save results
            result = {'labels': prediction, 'filename': sample['filename']}
            filename = os.path.join(output_dir, '%06d.mat' % i)
            print(filename)
            scipy.io.savemat(filename, result, do_compression=True)

        # measure elapsed time
        batch_time = time.time() - end
        print('[%d/%d], batch time %.2f' % (i, epoch_size, batch_time))

    # sum the values with same keys
    print('========================================================')
    result = {}
    num = len(metrics_all)
    print('%d images' % num)
    print('========================================================')
    for metrics in metrics_all:
        for k in metrics.keys():
            result[k] = result.get(k, 0) + metrics[k]

    for k in sorted(result.keys()):
        result[k] /= num
        print('%s: %f' % (k, result[k]))

    print('%.6f' % (result['Objects Precision']))
    print('%.6f' % (result['Objects Recall']))
    print('%.6f' % (result['Objects F-measure']))
    print('%.6f' % (result['Boundary Precision']))
    print('%.6f' % (result['Boundary Recall']))
    print('%.6f' % (result['Boundary F-measure']))
    print('%.6f' % (result['obj_detected_075_percentage']))

    print('========================================================')
    print(result)
    print('====================Refined=============================')

    result_refined = {}
    for metrics in metrics_all_refined:
        for k in metrics.keys():
            result_refined[k] = result_refined.get(k, 0) + metrics[k]

    for k in sorted(result_refined.keys()):
        result_refined[k] /= num
        print('%s: %f' % (k, result_refined[k]))
    print(result_refined)
    print('========================================================')


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Unseen Clustering Network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--dataset', dest='dataset_name',
                        required=True,
                        help='dataset to train on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Whether to ask user for confirmation before important steps and VIZ things",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to shuffle data",
    )    
    

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    # dataset
    dataset = get_dataset(args.dataset_name)
    worker_init_fn = dataset.worker_init_fn if hasattr(dataset, 'worker_init_fn') else None
    num_workers = 1
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=args.shuffle,
        num_workers=num_workers, worker_init_fn=worker_init_fn)
    print('Use dataset `{:s}` for training'.format(dataset.name))

    # prepare network
    gdino = GroundingDINOObjectPredictor()
    SAM = SegmentAnythingPredictor()

    # output dir
    output_dir = 'results/' + dataset._name
    print('Output will be saved to `{:s}`'.format(output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # test network
    test_segnet(dataloader, gdino, SAM, output_dir, args.vis)
