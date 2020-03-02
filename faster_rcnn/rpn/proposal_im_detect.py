"""
functions/rpn/proposal_im_detect.m
"""
import numpy as np
import scipy

from faster_rcnn.utils.blob import get_image_blob, get_image_blob_scales, \
    get_rois_blob
from faster_rcnn.rpn.proposal_locate_anchors import proposal_locate_anchors
from faster_rcnn.functions.fast_rcnn_bbox_transform_inv import fast_rcnn_bbox_transform_inv


def filter_boxes(min_box_size, boxes, scores):
    widths = boxes[:, 2] - boxes[:, 0] + 1
    heights = boxes[:, 3] - boxes[:, 1] + 1

    # valid_ind = widths >= min_box_size & heights >= min_box_size;
    a1 = widths >= min_box_size
    a2 = heights >= min_box_size
    valid_ind = a1 & a2
    valid_ind = valid_ind.transpose()

    boxes = boxes[np.where(valid_ind),][0]
    scores = scores[np.where(valid_ind),][0]

    return boxes, scores


def clip_boxes(boxes, im_width, im_height):
    boxes.T[0::4] = np.maximum(np.minimum(boxes.T[0::4].T, im_width), 1).T
    boxes.T[1::4] = np.maximum(np.minimum(boxes.T[1::4].T, im_height), 1).T
    boxes.T[2::4] = np.maximum(np.minimum(boxes.T[2::4].T, im_width), 1).T
    boxes.T[3::4] = np.maximum(np.minimum(boxes.T[3::4].T, im_height), 1).T
    return boxes


def proposal_im_detect(conf, caffe_net, im):
    im = scipy.single(im)
    im_blob, im_scales = get_image_blob(conf, im)

    im_size = [scipy.size(im, 0), scipy.size(im, 1)]
    scaled_im_size = list(np.round(np.array(im_size) * im_scales))
    data = np.array(im_blob)
    r, g, b = data.T
    im_blob2 = np.array([b, g, r])
    im_blob = im_blob2.transpose()
    im_blob = im_blob.transpose()

    im_blob = im_blob.transpose(1, 2, 0)
    im_blob = scipy.single(im_blob)
    im_blob = np.expand_dims(im_blob, axis=0)  # -> (1, 800, 600, 3)
    im_blob = im_blob.transpose(0, 3, 2, 1)  # -> (1, 3, 600, 800)

    net_inputs = im_blob

    caffe_net.reshape_as_input(net_inputs)
    output_blobs = caffe_net.forward(data=net_inputs)

    proposal_bbox_pred = output_blobs['proposal_bbox_pred'].transpose(0, 3, 2, 1)[0]
    box_deltas = proposal_bbox_pred
    featuremap_size = (scipy.size(box_deltas, 1), scipy.size(box_deltas, 0))
    box_deltas = box_deltas.reshape(-1, 4)

    conf = conf[0][0]

    anchors, _ = proposal_locate_anchors(conf, im.shape, conf['test_scales'], featuremap_size);
    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas)

    part1 = pred_boxes - 1
    part2 = np.subtract([im_size[1], im_size[0], im_size[1], im_size[0]], 1) / \
            np.subtract(
                [scaled_im_size[1], scaled_im_size[0], scaled_im_size[1], scaled_im_size[0]], 1)
    pred_boxes = np.multiply(part1, part2) + 1
    pred_boxes = clip_boxes(pred_boxes, im.shape[1], im.shape[0])

    scores = output_blobs['proposal_cls_prob'][0][-1::][0].T
    scores = scores.reshape(proposal_bbox_pred.shape[0], proposal_bbox_pred.shape[1], -1, order='F')
    scores = scores.transpose(2, 1, 0)

    scores = scores.flatten(1)

    pred_boxes, scores = filter_boxes(conf['test_min_box_size'], pred_boxes, scores)
    scores_ind = scores.argsort()[::-1]
    scores = np.sort(scores)[::-1]
    pred_boxes = pred_boxes[scores_ind,]

    return pred_boxes, scores

