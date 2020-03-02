import os
import sys

import math
import cv2
import caffe

sys.path.insert(0, '/app')
from faster_rcnn import Faster_Rcnn

caffe_root = "/opt/caffe/python"
MODEL_PATH = "./models/faster_rcnn_final"
sys.path.insert(0, caffe_root)

def main():

    p_input = sys.argv[1]
    p_model = sys.argv[2]
    try:
        p_output = sys.argv[3]
    except IndexError:
        p_output = 'boxed.png'
        print('No output file provided, will save as {}'.format(p_output))

    model_path = os.path.join(MODEL_PATH, p_model)

    print('[*] initializing model from {}'.format(model_path))
    fc = Faster_Rcnn(model_path)

    print('[*] detection state')
    classes, boxes_cell = fc.detect(p_input)

    print('[*] generating preview')
    cvimg = cv2.imread(p_input)
    for i in xrange(len(classes)):
        if not boxes_cell[i]:
            continue
        for bbox in boxes_cell[i]:
            startX, startY, endX, endY, confidence = bbox
            # print bbox
            cv2.rectangle(cvimg, (startX, startY), (endX, endY), (0, 255, 0), 1)
            cv2.putText(cvimg, str(confidence), (startX, startY), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0))
            class_name = classes[i]
            print('[*] detected object: {}, cors: {},{} {},{}'.format(class_name, startX,startY, endX, endY))
            cv2.putText(cvimg, class_name, (startX, int(endY - 40)), cv2.FONT_HERSHEY_PLAIN, 1.0,
                        (0, 255, 0))

    # cv2.imshow("boxed", cvimg)
    # cv2.waitKey(0)
    print('[*] saving preview output image as {}'.format(p_output))
    cv2.imwrite(p_output, cvimg)


if __name__ == "__main__":
    main()
