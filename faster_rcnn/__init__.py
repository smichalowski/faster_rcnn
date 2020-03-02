import os

import numpy as np
import scipy.io
import caffe

from faster_rcnn.rpn.proposal_im_detect import proposal_im_detect
from faster_rcnn.utils.nms import nms
from faster_rcnn.functions.fast_rcnn.fast_rcnn_conv_feat_detect import fast_rcnn_conv_feat_detect

def boxes_filter(aboxes, per_nms_topN, nms_overlap_thres, after_nms_topN):
    """experiments/script_faster_rcnn_demo.m"""

    if per_nms_topN > 0:
        aboxes = aboxes[0:min(len(aboxes), per_nms_topN), :]

    if (nms_overlap_thres > 0) & (nms_overlap_thres <= 1):
        aboxes = aboxes[nms(aboxes, nms_overlap_thres)]

    if after_nms_topN > 0:
        aboxes = aboxes[0:min(len(aboxes), after_nms_topN), :]

    return aboxes

class CaffeNet(caffe.Net):

    def reshape_as_input(self, input_data):

        for n in xrange(len(input_data)):
            if input_data[n].size == 0:
                continue
            input_data_size = input_data[n].shape  # [::-1]
            input_data_size_extended = scipy.ones(4 - len(input_data_size),
                                                  dtype=np.int).tolist() + list(
                input_data_size)
            _idx = self.inputs[n]
            self.blobs[_idx].reshape(*input_data_size_extended)  # 1,3,600,800
            self.reshape()


class Faster_Rcnn():

    # nms_overlap_tresh make greater than 0.7 for more boxes proposals
    opts = {
        'per_nms_topN': 6000,
        'nms_overlap_thresh': 0.7,
        'after_nms_topN': 300,
        'test_scales': 600
    }

    def __init__(self, model_path, use_gpu_id=None):

        matlab_model_path = os.path.join(model_path, 'model.mat')
        ld = scipy.io.loadmat(matlab_model_path)
        self._proposal_detection_model = ld['proposal_detection_model'];
        fn_proposal_net_def = self._proposal_detection_model['proposal_net_def'][0][0][0]
        fn_proposal_net = self._proposal_detection_model['proposal_net'][0][0][0]
        fn_detection_net_def = self._proposal_detection_model['detection_net_def'][0][0][0]
        fn_detection_net = self._proposal_detection_model['detection_net'][0][0][0]
        proposal_net_def = str(os.path.join(model_path, fn_proposal_net_def))
        proposal_net = str(os.path.join(model_path, fn_proposal_net))
        detection_net_def = str(os.path.join(model_path, fn_detection_net_def))
        detection_net = str(os.path.join(model_path, fn_detection_net))

        self._proposal_detection_model['conf_proposal'][0][0][0]['test_scales'] = \
            self.opts['test_scales']
        self._proposal_detection_model['conf_detection'][0][0][0]['test_scales'] = \
            self.opts['test_scales']

        # proposal net
        self._rpn_net = CaffeNet(proposal_net_def, caffe.TEST)
        self._rpn_net.copy_from(proposal_net)
        self._fast_rcnn_net = CaffeNet(detection_net_def, caffe.TEST)
        self._fast_rcnn_net.copy_from(detection_net)

        if use_gpu_id is None:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(use_gpu_id)

        print('[*] warming up....')
        for i in xrange(2):
            print("- step {}".format(i))

            im = scipy.uint8(scipy.ones((375, 500, 3), scipy.uint8) * 128)
            boxes, scores = proposal_im_detect(self._proposal_detection_model['conf_proposal'],
                                               self._rpn_net, im)

            boxes_and_scores = np.concatenate((boxes, np.expand_dims(scores, axis=1)), axis=1)
            aboxes = boxes_filter(boxes_and_scores, self.opts['per_nms_topN'],
                                  self.opts['nms_overlap_thresh'],
                                  self.opts['after_nms_topN'])

            _is_share_feature = self._proposal_detection_model['is_share_feature'][0][0][0].astype(bool)[0]
            _last_shared_output_blob_name = \
            self._proposal_detection_model['last_shared_output_blob_name'][0][0][0]
            _4d_aboxes = aboxes.T[:-1, :].T  # we need only 4 dimensions

            if _is_share_feature:
                boxes, scores = fast_rcnn_conv_feat_detect(
                    self._proposal_detection_model['conf_detection'], \
                    self._fast_rcnn_net, im, self._rpn_net.blobs[_last_shared_output_blob_name],
                    _4d_aboxes, self.opts['after_nms_topN'])
            else:
                raise NotImplementedError()

        print('[*] warming up completed !')


    def detect(self, image_path):
        img = scipy.misc.imread(image_path)
        boxes, scores = proposal_im_detect(self._proposal_detection_model['conf_proposal'], self._rpn_net, img)
        boxes_and_scores = np.concatenate((boxes, np.expand_dims(scores, axis=1)), axis=1)
        aboxes = boxes_filter(boxes_and_scores, self.opts['per_nms_topN'], self.opts['nms_overlap_thresh'],
                              self.opts['after_nms_topN'])

        _is_share_feature = self._proposal_detection_model['is_share_feature'][0][0][0].astype(bool)[0]
        _last_shared_output_blob_name = self._proposal_detection_model['last_shared_output_blob_name'][0][0][
            0]
        _4d_aboxes = aboxes.T[:-1, :].T  # we need only 4 dimensions

        if _is_share_feature:
            boxes, scores = fast_rcnn_conv_feat_detect( \
                self._proposal_detection_model['conf_detection'], \
                self._fast_rcnn_net, img, self._rpn_net.blobs[_last_shared_output_blob_name], \
                _4d_aboxes, self.opts['after_nms_topN'])
            pass
        else:
            raise NotImplementedError()

        classes = self._proposal_detection_model['classes'][0][0]
        boxes_cell = []
        thres = 0.6

        for i in xrange(len(classes)):
            _i = i + 1
            p1 = boxes.T[(1 + (_i - 1) * 4) - 1:(_i * 4)].T
            p2 = scores.T[i].T  # (300,)
            p2 = np.expand_dims(p2, 1)  # -> (300,1)
            boxes_cell.insert(i, np.concatenate((p1, p2), axis=1))
            boxes_cell[i] = boxes_cell[i][nms(boxes_cell[i], 0.3)]
            I = boxes_cell[i].T[4] >= thres
            boxes_cell[i] = boxes_cell[i][I]

        # remove data left by caffe
        classes = [str(c[0][0]) if c.any() else '' for c in classes]
        boxes_cell = [[list(bc2) for bc2 in bc] if bc.any() else '' for bc in boxes_cell]
        return classes, boxes_cell
