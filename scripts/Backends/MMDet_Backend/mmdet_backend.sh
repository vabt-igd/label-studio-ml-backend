#!/bin/bash
BACKEND_CODE_PATH=./label_studio_ml/examples/mmdetection-3
BACKEND_BUILD_PATH=./mmDet_backend
BACKEND_SCRIPT=mmdetection.py

echo " 
===================================================
Cloning the Label Studio ML Backend repository...
 "
git clone https://github.com/vabt-igd/label-studio-ml-backend.git
cd label-studio-ml-backend
git pull


echo " 
===================================================
Creating a virtual environment using venv and installing base dependencies...
 "
python -m venv ml-backend-venv
source ml-backend-venv/Scripts/activate # assuming you're on bash or zsh
pip install -U pip
pip install -U -e .


echo " 
===================================================
Installing backend depedencies...
 "
pip install -r $BACKEND_CODE_PATH/requirements.txt
mim install mmengine
mim download mmdet --config yolov3_mobilenetv2_8xb24-320-300e_coco --dest $BACKEND_CODE_PATH/
mim install mmcv==2.0.0
export checkpoint_file=$BACKEND_CODE_PATH/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth
export config_file=$BACKEND_CODE_PATH/yolov3_mobilenetv2_8xb24-320-300e_coco.py


echo " 
===================================================
Initializing backend...
 "
rm -rf $BACKEND_BUILD_PATH
label-studio-ml init $BACKEND_BUILD_PATH --script $BACKEND_CODE_PATH/$BACKEND_SCRIPT


echo " 
===================================================
Starting the backend...
 "
# label-studio-ml start $BACKEND_BUILD_PATH
cd $BACKEND_BUILD_PATH
python _wsgi.py -p 1716