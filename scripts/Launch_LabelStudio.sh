#!/bin/bash

echo " 
=====================================
Installing / Updating Label Studio...
"
# Install the package
# into python virtual environment
python -m pip install -U label-studio


echo " 
=====================================
"
Launching Label Studio...
# Launch it!
label-studio