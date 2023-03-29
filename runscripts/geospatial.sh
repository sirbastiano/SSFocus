#!/bin/bash

# Set environment name and Python version
ENV_NAME="geospatial"
PYTHON_VERSION="3.9"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install conda before running this script."
    exit 1
fi

echo "Creating a new conda environment named '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create --name $ENV_NAME python=$PYTHON_VERSION -y

# Check if conda environment was created successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to create conda environment. Exiting."
    exit 1
fi

echo "Activating the new environment '$ENV_NAME'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Check if conda environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment. Exiting."
    exit 1
fi

echo "Installing geospatial libraries..."
conda install -c conda-forge rasterio pandas xarray opencv geopandas fiona shapely pyproj cartopy gdal matplotlib seaborn jupyterlab -y

# Check if geospatial libraries were installed successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to install geospatial libraries. Exiting."
    exit 1
fi

echo "Installation of geospatial libraries complete. Use 'conda activate $ENV_NAME' to activate the environment."
