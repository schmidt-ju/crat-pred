#!/bin/bash

# Folder structure
mkdir -p dataset/argoverse
cd dataset/argoverse

# Download
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_train_v1.1.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_val_v1.1.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/forecasting_test_v1.1.tar.gz
wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz

# Extract
tar xvf forecasting_train_v1.1.tar.gz
tar xvf forecasting_val_v1.1.tar.gz
tar xvf forecasting_test_v1.1.tar.gz
tar xf hd_maps.tar.gz

# Argoverse needs the map_files folder
SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -r map_files $SITE_PACKAGES