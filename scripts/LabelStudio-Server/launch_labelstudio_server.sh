#!/bin/bash

echo "
===================================================
Creating a virtual environment using venv and installing base dependencies...
 "
python -m venv labelstudio-venv
source ml-backend-venv/Scripts/activate # assuming you're on bash or zsh
pip install -U pip


echo "
===================================================
Installing / Updating Label Studio...
"
pip install -U label-studio


echo "
===================================================
Launching Label Studio...
"
label-studio