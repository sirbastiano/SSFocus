# Define the filename for the configuration file
CONFIG_FILE="config.ini"

# Get the current directory path
CURRENT_DIR=$(pwd) || { echo "Error: Cannot get the current directory path." >&2; exit 1; }

# Write the configuration to the file
cat <<EOF > $CONFIG_FILE
[DATABASE]
DB_NAME = SARLens_db
DB_USER = myuser # This is the user that has access to the ASF database
DB_PASSWORD = mypassword # This is the password of the user

[LOGGING]
LOG_LEVEL = INFO

[DIRECTORIES]
SARLENS_DIR = $CURRENT_DIR
EOF

# Print a message indicating that the configuration file has been created
echo "Configuration file created at $(date) at $PWD."