#!/bin/bash
# redirect stdout/stderr to a file
# remove logfile if existing:
rm -f logfile_decoding.log
exec >logfile_decoding.log 2>&1

decode_files() {
    Now1=$(date +"%Y-%m-%d %H:%M:%S")
    echo "========= Start Decoding ========= $Now1"
    echo "Input product is:" $1
    echo "Output directory is:" $2
    python -m SARProcessor.decode --inputfile $1 --output $2
    Now2=$(date +"%Y-%m-%d %H:%M:%S")
    echo "========= End Decoding ========= $Now2"
    # computer time difference and display it
}

decode_dat_files() {
    datfiles=$(ls /home/roberto/PythonProjects/SSFocus/Data/RAW/SM/dat/*.dat)

    for datfile in $datfiles; do
        # get the filename without extension
        datfile_basename=$(basename -- "$datfile" .dat)
        npyfile="/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy/$datfile_basename.npy"

        # check if the npy file exists
        if [[ ! -f $npyfile ]]; then
            echo "WARNING: $datfile_basename has not been decoded to npy"
            decode_files "$datfile" "/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/numpy"
        fi
    done
}

clear
# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Activate the SARLens environment
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


# check if all products have been decoded
# list the files in output_dir and check which one is missing from zipfiles
# the decoded files have the same name as the zipfiles but with .npy extension
counter=0
while [ $counter -lt 1 ]
do
    decode_dat_files
    counter=$((counter+1))
done


# terminate the script
exit 0
