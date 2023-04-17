#!/bin/bash
BACKEND_CODE_PATH=./label_studio_ml/examples/sam_segmentation
BACKEND_BUILD_PATH=./sam_backend
BACKEND_SCRIPT=sam_backend.py

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
python _wsgi.py -p 1717