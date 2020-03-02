"""
functions/fast_rcnn_bbox_transform_inv.m
"""
import numpy as np

def fast_rcnn_bbox_transform_inv(boxes, box_deltas):

    src_w = boxes[:, 2:3] - boxes[:, 0:1] + 1
    src_h = boxes[:, 3:4] - boxes[:, 1:2] + 1
    src_ctr_x = boxes[:, 0:1] + 0.5 * (src_w - 1)
    src_ctr_y = boxes[:, 1:2] + 0.5 * (src_h - 1)

    dst_ctr_x = box_deltas.T[0::4].T
    dst_ctr_y = box_deltas.T[1::4].T
    dst_scl_x = box_deltas.T[2::4].T
    dst_scl_y = box_deltas.T[3::4].T

    pred_ctr_x = np.add(np.multiply(dst_ctr_x, src_w), src_ctr_x)
    pred_ctr_y = np.add(np.multiply(dst_ctr_y, src_h), src_ctr_y)

    pred_w = np.multiply(np.exp(dst_scl_x), src_w)
    pred_h = np.multiply(np.exp(dst_scl_y), src_h)

    pred_boxes = np.zeros(box_deltas.shape, dtype=np.float32)
    # ponizsze dzialalo na rpmach, ale wywalalo sie w fast_rcnn
    # pred_boxes[:,:1] = np.subtract(pred_ctr_x, 0.5*(pred_w-1))
    # pred_boxes[:,1:2] = np.subtract(pred_ctr_y, 0.5*(pred_h-1))
    # pred_boxes[:,2:3] = np.add(pred_ctr_x, 0.5*(pred_w-1))
    # pred_boxes[:,3:4] = np.add(pred_ctr_y, 0.5*(pred_h-1))
    pred_boxes.T[0::4] = np.subtract(pred_ctr_x, 0.5 * (pred_w - 1)).T
    pred_boxes.T[1::4] = np.subtract(pred_ctr_y, 0.5 * (pred_h - 1)).T
    pred_boxes.T[2::4] = np.add(pred_ctr_x, 0.5 * (pred_w - 1)).T
    pred_boxes.T[3::4] = np.add(pred_ctr_y, 0.5 * (pred_h - 1)).T

    return pred_boxes

