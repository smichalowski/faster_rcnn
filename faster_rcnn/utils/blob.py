import scipy
import cv2
import numpy as np


def prep_im_for_blob(im, im_means, target_size, max_size):
    im = scipy.single(im)

    im_means4 = cv2.resize(im_means, (scipy.size(im, 1), scipy.size(im, 0)),
                            interpolation=cv2.INTER_LINEAR)
    im_means = im_means4
    im = im - im_means
    im_scale = prep_im_for_blob_size((scipy.size(im, 0),
        scipy.size(im, 1)), target_size, max_size)
    target_size = (int(scipy.size(im, 1) * im_scale),
        int(scipy.size(im, 0) * im_scale))
    im1 = cv2.resize(im, target_size, interpolation=cv2.INTER_LINEAR)

    return im1, im_scale

def prep_im_for_blob_size(im_size, target_size, max_size):
    im_size_min = min(im_size[0:2])
    im_size_max = max(im_size[0:2])
    im_scale = float(target_size) / im_size_min
    if round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    return im_scale

def im_list_to_blob(ims):
    """Convert a list of images into a network input.
    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
        dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def get_rois_blob(conf, im_rois, im_scale_factors):
    feat_rois, levels = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors)
    rois_blob = np.concatenate((levels.T, feat_rois.T)).T
    rois_blob = scipy.single(rois_blob)
    return rois_blob

def get_image_blob(conf, im):
    image_means = conf[0][0][0]['image_means'][0]
    test_scales = int(conf[0][0][0]['test_scales'])
    test_max_size = int(conf[0][0][0]['test_max_size'])
    if True:
        blob, im_scales = prep_im_for_blob(im, image_means, test_scales, test_max_size)
    return blob, im_scales

def get_image_blob_scales(conf, im):
    im_scales = prep_im_for_blob_size(im.shape, conf['test_scales'], conf['test_max_size'])
    return im_scales

def get_blobs(conf, im, rois):
    im_scale_factors = get_image_blob_scales(conf, im)
    rois_blob = get_rois_blob(conf, rois, im_scale_factors)
    return rois_blob, im_scale_factors

def map_im_rois_to_feat_rois(conf, im_rois, scales):
    im_rois = scipy.single(im_rois)
    levels = scipy.ones(im_rois.shape[0])
    levels = np.expand_dims(levels, 1)

    # FIXME: below
    __scales = np.array(levels)
    __scales.fill(scales)
    bsxfunret = ((im_rois - 1) * __scales) + 1
    feat_rois = np.around(bsxfunret)

    return feat_rois, levels
