import numpy as np

from faster_rcnn.utils.blob import prep_im_for_blob_size

def proposal_locate_anchors(conf, im_size, target_scale, feature_map_size):
    # % only for fcn
    # if ~exist('feature_map_size', 'var')
    #     feature_map_size = [];
    # end

    if target_scale:
        anchors, im_scales = proposal_locate_anchors_single_scale(im_size, conf, target_scale,
                                                                  feature_map_size)
    else:
        raise NotImplementedError

    return anchors, im_scales


def proposal_locate_anchors_single_scale(im_size, conf, target_scale, feature_map_size):
    if not feature_map_size:
        raise NotImplementedError
    else:
        im_scale = prep_im_for_blob_size(im_size, target_scale, conf['max_size'])
        output_size = feature_map_size

    shift_x = np.arange(output_size[1]) * conf['feat_stride'].astype(int).item()
    shift_y = np.arange(output_size[0]) * conf['feat_stride'].astype(int).item()
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # % concat anchors as [channel, height, width], where channel is the fastest dimension.
    # anchors = reshape(bsxfun(@plus, permute(conf.anchors, [1, 3, 2]), ...
    #    permute([shift_x(:), shift_y(:), shift_x(:), shift_y(:)], [3, 1, 2])), [], 4);
    z = np.transpose(np.expand_dims(conf['anchors'][0][0], axis=0), (1, 0, 2))
    y = np.array((shift_x.flatten(1), shift_y.flatten(1), shift_x.flatten(1),
                  shift_y.flatten(1))).transpose()
    y = np.expand_dims(y, axis=0)

    anchors = y + z

    anchors = anchors.reshape(-1, 4, order='F')

    return anchors, im_scale
