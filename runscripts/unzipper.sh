#!/bin/bash

# Folder where the zip files are located
ZIP_FOLDER="/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/L0"
DAT_FOLDER="/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/DAT"


# Folder where the unzipped contents should be placed
UNZIP_FOLDER="/Users/robertodelprete/Desktop/PyScripts/SARLens/SSFocus/Data/RAW/unzipped"

# Check if the ZIP_FOLDER exists
if [ ! -d "$ZIP_FOLDER" ]; then
    echo "The folder $ZIP_FOLDER does not exist."
    exit 1
fi

# Check if the UNZIP_FOLDER exists, create it if it doesn't
if [ ! -d "$UNZIP_FOLDER" ]; then
    mkdir -p "$UNZIP_FOLDER"
fi

# Loop through each zip file and unzip it
for zip_file in "$ZIP_FOLDER"/*.zip; do
    # Extract just the filename without the full path
    file_name=$(basename "$zip_file")
    
    # Create a directory based on the filename
    unzip_dir="$UNZIP_FOLDER/${file_name%.*}"
    
    # Create the directory if it doesn't exist
    if [ ! -d "$unzip_dir" ]; then
        mkdir -p "$unzip_dir"
    fi
    
    # Unzip the file
    unzip -o "$zip_file" -d "$unzip_dir"
done

# echo "All files have been unzipped."

# Check if DAT_FOLDER exists, create if it doesn't
if [ ! -d "$DAT_FOLDER" ]; then
    mkdir -p "$DAT_FOLDER"
fi

# Loop through each folder in unzip_dir
for fold in "$UNZIP_FOLDER"/*; do
    # Check if it's a directory
    if [ -d "$fold" ]; then
        # Find all .dat files that don't contain the words "annot" or "index"
        found_files=$(find "$fold" -type f -name "*.dat" ! -name "*annot*" ! -name "*index*")
        # Count the number of lines (i.e., found files)
        num_files=$(echo "$found_files" | grep -c '.*')
        echo "Found num: $num_files files in $(basename "$fold")"
        # copy files to DAT_FOLDER
        cp -r "$num_files" "$DAT_FOLDER"
    fi
done

