
INPUT_PRODUCT=/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/Extracted/S1A_S1_RAW__0SSV_20200110T183210_20200110T183241_030742_038657_8CCA.SAFE/s1a-s1-raw-s-vv-20200110t183210-20200110t183241-030742-038657.dat
OUTPUT_DIR=/home/roberto/PythonProjects/SSFocus/Data/FOCUSED/SM/S1A_S1_RAW__0SSV_20200110T183210_20200110T183241_030742_038657_8CCA

echo "Focusing product $INPUT_PRODUCT"

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Activate the AINavi environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate SARLens; then
    echo "SARLens environment activated"
else
    echo "SARLens environment not found"
    exit 1
fi

# Check if the script has been sourced
if [ -z "$BASH_SOURCE" ]; then
    echo "ERROR: You must source this script. Run 'source $0'"
    exit 1
fi
# Check if the script is being run from the right directory
if [ ! -f setup.py ]; then
  echo "ERROR: Run this from the top-level directory of the repository"
  exit 1
fi

# check if output directory exists, else create it:
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

echo "========= Start focus ========="
python -m SARProcessor.focus --input $INPUT_PRODUCT --output $OUTPUT_DIR
echo "========= End focus ========="

