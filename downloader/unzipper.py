# write a script to unzip the downloaded file
# the zip files are in Data/RAW 
# each filename starts with "S1"

import os
import zipfile
from pathlib import Path

# get the current working directory
cwd = os.getcwd()
# get the path to the zip files
zip_dir = os.path.join(cwd, "Data", "RAW")
zipFiles = [x for x in Path(zip_dir).iterdir() if x.is_file() and x.name.startswith("S1")]

for f in zipFiles:
    with zipfile.ZipFile(f, 'r') as zip_ref:
        zip_ref.extractall(zip_dir)
    # remove the zip file
    os.remove(f)
    


