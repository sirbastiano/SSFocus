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

INPUT_PRODUCT=/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/S1A_S1_SLC__1SSV_20200930T183218_20200930T183248_034592_0406FD_1E25.zip
# use the same folder of INPUT_PRODUCT, split the name:
OUTPUT_DIR=$(dirname $INPUT_PRODUCT)/$(basename $INPUT_PRODUCT .zip)
echo "Output directory is:" $OUTPUT_DIR


echo "Unzipping product $INPUT_PRODUCT"
unzip $INPUT_PRODUCT -d $OUTPUT_DIR 
# delete the zipfile:
rm $INPUT_PRODUCT
echo "Unzipping done, deleted $INPUT_PRODUCT"

echo "Focusing product $INPUT_PRODUCT"
FOCUS_DIR=/home/roberto/PythonProjects/SSFocus/Data/RAW/FOCUSED
# # check if output directory exists, else create it:
# if [ ! -d "$OUTPUT_DIR" ]; then
#   mkdir -p "$OUTPUT_DIR"
# fi

# echo "========= Start focus ========="
# python -m SARProcessor.focus --input $INPUT_PRODUCT --output $OUTPUT_DIR
# echo "========= End focus ========="

