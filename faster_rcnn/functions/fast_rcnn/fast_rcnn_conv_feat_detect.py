"""
functions/fast_rcnn/fast_rcnn_conv_feat_detect.m
"""
import math

import scipy
import numpy as np

from faster_rcnn.functions.fast_rcnn_bbox_transform_inv import fast_rcnn_bbox_transform_inv
from faster_rcnn.rpn.proposal_im_detect import clip_boxes
from faster_rcnn.utils.blob import get_blobs


def fast_rcnn_conv_feat_detect(conf, caffe_net, im, conv_feat_blob, boxes,
    max_rois_num_in_gpu):

    conf = conf[0][0][0]

    rois_blob, _ = get_blobs(conf, im, boxes)
    rois_blob = rois_blob - 1

    rois_blob = np.expand_dims(np.expand_dims(rois_blob.T, 0), 0)
    rois_blob = scipy.single(rois_blob)

    caffe_net.blobs['data'].reshape(*conv_feat_blob.data.shape)
    np.copyto(caffe_net.blobs['data'].data, conv_feat_blob.data)

    total_rois = scipy.size(rois_blob, 3)

    _pseudo_magic = int(math.ceil(float(total_rois) / max_rois_num_in_gpu))
    total_scores = []  # np.array(_pseudo_magic)
    total_box_deltas = []  # np.array(_pseudo_magic)

    for i in xrange(_pseudo_magic):

        _i = i + 1
        sub_ind_start = 1 + (_i - 1) * max_rois_num_in_gpu
        sub_ind_end = min(total_rois, _i * max_rois_num_in_gpu)
        # MATLAB: sub_rois_blob = rois_blob(:, :, :, sub_ind_start:sub_ind_end);
        sub_rois_blob = rois_blob.T[sub_ind_start - 1:sub_ind_end, ]

        net_inputs = [np.ndarray(0), sub_rois_blob]

        caffe_net.reshape_as_input(net_inputs)
        output_blobs = caffe_net.forward(rois=sub_rois_blob, data=conv_feat_blob.data)

        if conf['test_binary']:
            # % simulate binary logistic regression
            raise Exception('Not implemented')
        else:
            # % use softmax estimated probabilities
            scores = output_blobs['cls_prob']
            scores = np.squeeze(scores)

        # % Apply bounding-box regression deltas
        box_deltas = output_blobs['bbox_pred']
        box_deltas = np.squeeze(box_deltas)

        total_scores.insert(i, scores)
        total_box_deltas.insert(i, box_deltas)

    pred_boxes = fast_rcnn_bbox_transform_inv(boxes, box_deltas)
    pred_boxes = clip_boxes(pred_boxes, im.shape[1], im.shape[0])

    pred_boxes = pred_boxes.T[4:].T
    scores = scores.T[1:].T

    return pred_boxes, scores

