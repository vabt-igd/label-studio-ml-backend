#!/bin/bash
BACKEND_CODE_PATH=./label_studio_ml/examples/segment_anything_model
VENV_FOLDER=./ls-sam-venv

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
python -m venv $VENV_FOLDER
source $VENV_FOLDER/Scripts/activate # assuming you're on bash or zsh
pip install -U pip 
pip install -U wheel setuptools
# pip install -U -e .


echo " 
===================================================
Installing backend depedencies...
 "

pip install -r $BACKEND_CODE_PATH/requirements.txt
SAM_CHOICE="SAM" # MobileSAM


echo " 
===================================================
Starting the backend...
 "
label-studio-ml start $BACKEND_CODE_PATH -p 1718