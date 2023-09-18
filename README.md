# RAW Downloader

This repo handles the automatic download of RAW SM Sentinel-1 data from the Copernicus Alaska Dataframe Facility (ASF) trough their [API](https://docs.asf.alaska.edu/asf_search/basics/). The data is downloaded in the SAFE format and then converted to the Level 0 format using the [sentinel1decoder](https://github.com/Rich-Hall/sentinel1decoder).


## Installation

Modify the config.sh file located in installer to reflect your username and password of ASF.

```
DB_USER = myuser # This is the user that has access to the ASF database
DB_PASSWORD = mypassword # This is the password of the user
```



To install the dependencies, simply use the following CLI command:

```
source installer/install.sh
```

## Support

For any inquiries, issues, or suggestions, please visit the SARLens GitHub repository and submit an issue or pull request.

