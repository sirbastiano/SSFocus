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
if conda activate SARLens; then
    echo "SARLens environment activated"
else
    echo "SARLens environment not found"
    exit 1
fi



numpy_dir="/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/NUMPY"


decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200403t183209-20200403t183240-031967-03b114.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200310t183209-20200310t183239-031617-03a4c7.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200322t183209-20200322t183240-031792-03aaec.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200415t183210-20200415t183240-032142-03b73e.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200427t183210-20200427t183241-032317-03bd63.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200509t183227-20200509t183238-032492-03c34a.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200825t183217-20200825t183247-034067-03f47f.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200906t183217-20200906t183248-034242-03faa1.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200918t183217-20200918t183248-034417-0400c7.dat" $numpy_dir
# decode_file "/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT/s1a-s1-raw-s-vv-20200930t183218-20200930t183248-034592-0406fd.dat" $numpy_dir

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
