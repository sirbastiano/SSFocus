function unzip_copy_remove() {
  local input_product=$1
  local dat_directory="/home/roberto/PythonProjects/SSFocus/Data/RAW/SM/dat/"

  local output_zip_dir=$(dirname "$input_product")/$(basename "$input_product" .zip)
  echo "Output directory is: $output_zip_dir"

  # Unzip and remove the product in the output directory
  echo "Unzipping product $input_product"
  unzip "$input_product" -d "$output_zip_dir"

  local focus_prod=$(find "$output_zip_dir" -type f -name "*.dat" ! -name "*index*" ! -name "*annot*")

  # Copy found .dat files to dat_directory
  cp "$focus_prod" "$dat_directory"
  # removing the product in the output directory:
  rm "$input_product"
  rm -rf "$output_zip_dir"
}

# Example usage
zipfiles=$(find "/home/roberto/PythonProjects/SSFocus/Data/RAW/L0/" -type f -name "*.zip")
for zipfile in $zipfiles; do
  unzip_copy_remove "$zipfile"
done
