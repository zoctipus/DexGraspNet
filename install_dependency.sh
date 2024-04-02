#!/bin/bash

# Navigate to thirdparty/ directory
cd thirdparty/

# Install pytorch_kinematics
cd pytorch_kinematics/
pip install -e .
cd ../

# Install pytorch3d
cd pytorch3d/
pip install -e .
cd ../

# Install TorchSDF
cd TorchSDF/
git checkout 0.1.0
bash install.sh
cd ../
