# write a script to unzip the downloaded file
# the zip files are in Data/RAW 
# each filename starts with "S1"
import os
import zipfile
from pathlib import Path
import logging

# set up logging
# Create a logger with the name "myapp"
logger = logging.getLogger("Unzipper")

# Set the logging level to INFO
logger.setLevel(logging.INFO)

# Create a file handler and set its logging level to INFO
file_handler = logging.FileHandler("unzip.log")
file_handler.setLevel(logging.INFO)

# Create a console handler and set its logging level to DEBUG
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Define a formatter for the handlers
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# get the current working directory
wordir = os.environ["SARLENS_DIR"]

# get the path to the zip files
zip_dir = os.path.join(wordir, "Data","RAW","IW")
zip_dir = Path(zip_dir)

def main():
    # get a list of all the zip files
    zipFiles = [x for x in zip_dir.glob('**/*.zip') if x.is_file() and x.name.startswith("S1")]
    assert len(zipFiles) > 0, f"No zip files found in {zip_dir}"
    logger.info(f"Found {len(zipFiles)} zip files in {zip_dir}")
    
    for f in zipFiles:
        logger.info (f"Unzipping: {f}")
        with zipfile.ZipFile(f, 'r') as zip_ref:
            logger  .info(f"Extracting {f} to {zip_dir}")
            zip_ref.extractall(zip_dir)
        
        # remove the zip file
        logger.info (f"Removing zipfile: {f}")
        os.remove(f)
        
if __name__ == "__main__":
    """ This is executed when run from the command line:
        python -m Downloader.unzipper
    """
    main()