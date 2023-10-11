import configparser
import os

# Load configuration variables from a file
config = configparser.ConfigParser()
# To launch from internal folders.
try:
    config.read("config.ini")
    SARLENS_DIR = config["DIRECTORIES"]["SARLENS_DIR"]
except KeyError:
    config.read("../config.ini")
    SARLENS_DIR = config["DIRECTORIES"]["SARLENS_DIR"]
    
    
# Set environment variables
os.environ["SARLENS_DIR"] = SARLENS_DIR
