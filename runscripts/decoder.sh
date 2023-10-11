#!/bin/bash
# redirect stdout/stderr to a file
# remove logfile if existing:
rm -f logfile_decoding.log
exec >logfile_decoding.log 2>&1

decode_file() {
    Now1=$(date +"%Y-%m-%d %H:%M:%S")
    echo "========= Start Decoding ========= $Now1"
    echo "Input product is:" $1
    echo "Output directory is:" $2
    python -m SARProcessor.decode --inputfile $1 --output $2
    Now2=$(date +"%Y-%m-%d %H:%M:%S")
    echo "========= End Decoding ========= $Now2"
    # computer time difference and display it
}

# decode_dat_files() {
#     datfiles=$(ls /Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/*.dat)

#     for datfile in $datfiles; do
#         # get the filename without extension
#         datfile_basename=$(basename -- "$datfile" .dat)
#         npyfile="/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/NUMPY/$datfile_basename.npy"


#         echo "Decoding $datfile_basename"
#         decode_file "$datfile"  "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/NUMPY"        

#         # check if the npy file exists
#         if [[ ! -f $npyfile ]]; then
#             echo "WARNING: $datfile_basename has not been decoded to npy"
#             decode_files $datfile "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/NUMPY"
#         fi
#     done
# }

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
if conda activate py311; then
    echo "SARLens environment activated"
else
    echo "SARLens environment not found"
    exit 1
fi



outdir="/media/warmachine/DBDISK/SSFocus/data/Raw_decoded"


decode_file "/media/warmachine/DBDISK/SSFocus/data/RAW/S1A_EW_RAW__0SDH_20231008T114011_20231008T114119_050673_061B0A_351B.SAFE/s1a-ew-raw-s-hh-20231008t114011-20231008t114119-050673-061b0a.dat" $outdir

# check if all products have been decoded
# list the files in output_dir and check which one is missing from zipfiles
# the decoded files have the same name as the zipfiles but with .npy extension
# counter=0
# while [ $counter -lt 1 ]
# do
#     decode_dat_files
#     counter=$((counter+1))
# done


# terminate the script
exit 0
