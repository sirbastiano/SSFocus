# Check if Conda is installed
clear
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

# Config:
OUTPUT_DIR="/home/roberto/PythonProjects/SSFocus/Data/RAW/FOCUSED"
RAW_DIR="/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy"
# Go in output dir, fetech .dat file not containing "index" or "annot" in the name:
data_products=$(find $RAW_DIR -type f -name "*.npy" ! -name "*meta*" ! -name "*ephemeris*")
meta_files=$(find $RAW_DIR -type f -name "*meta*")
ephemeris_files=$(find $RAW_DIR -type f -name "*ephemeris*")


# Focus all the products in Raw_products:
echo "========= Start focus ========="
for product in ${data_products[@]}; do
    echo "Input product is:" $product
    echo "Output directory is:" $OUTPUT_DIR
    # Get meta as product basename + "_pkt_8_metadata.pkl"
    meta=$(echo $product | sed 's/.npy/_pkt_8_metadata.pkl/g')
    echo "Meta file is:" $meta
    # Get ephemeris as product basename + "_ephemeris.pkl"
    ephemeris=$(echo $product | sed 's/.npy/_ephemeris.pkl/g')
    echo "Ephemeris file is:" $ephemeris


    idx_chunk=1
    echo "python -m SARProcessor.focus --data $product --meta $meta --ephemeris $ephemeris --output $OUTPUT_DIR --backend "numpy""
    # python -m SARProcessor.focus --data $product --meta $meta --ephemeris $ephemeris --output $OUTPUT_DIR --backend "numpy" --idx_chunk $idx_chunk
done
echo "========= End focus ========="

