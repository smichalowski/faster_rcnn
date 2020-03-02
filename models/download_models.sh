#!/bin/sh
wget 'https://onedrive.live.com/download?resid=D7AF52BADBA8A4BC!114&authkey=!AERHoxZ-iAx_j34&ithint=file%2czipfaster_rcnn_final_model.zip' -O faster_rcnn_final_model.zip
unzip faster_rcnn_final_model.zip
mv output/faster_rcnn_final .
rm -rf output
rm faster_rcnn_final_model.zip
