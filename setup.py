from setuptools import setup, find_packages
import os
import subprocess
import platform

def is_linux():
    print("Platform system: ",platform.system())
    return platform.system() == 'Linux'

if is_linux():
    print("The system is running on Linux.")
    system = 0
    
else:
    print("The system is not running on Linux.")
    system = 1

def install_dependencies():
    torch_dependencies = "https://download.pytorch.org/whl/torch_stable.html"

    if system == 0:
        subprocess.check_call(["pip", "install", "torch==2.0.0+cu118", "torchvision==0.15.1+cu118", "-f", torch_dependencies])
    else:
        subprocess.check_call(["pip", "install", "torch==2.0.0", "torchvision==0.15.1", "-f", torch_dependencies])

install_dependencies()

setup(
    name='SARLens',
    version='0.1',
    description='SAR Focusing using AI',
    author='Roberto Del Prete',
    author_email='roberto.delprete@ext.esa.int',
    packages=find_packages(),
    install_requires=[
              'h5py',
              'netcdf4',
              'h5netcdf',
              'rasterio',
              'rioxarray',
              'numpy',
              'scikit-image',
              'scipy',
              'scikit-learn',
              'xarray',
              'geopandas',
              'pandas>=1.4, <2',
              'asf_search',
              'matplotlib',
              'seaborn',
              'tqdm',
              'tzlocal',
              'regex',
       ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

