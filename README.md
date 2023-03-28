# SARLens: Synthetic Aperture Radar (SAR) Image Focusing Library

## Introduction

**SARLens** is a cutting-edge, avant-garde Python package designed for the purpose of decoding and focusing raw Sentinel-1 Synthetic Aperture Radar (SAR) images through the implementation of pioneering deep learning techniques. The library expertly processes the satellite data, employing the Range-Doppler algorithm for optimal image focusing. SARLens stands as an instrumental tool for remote sensing experts, computer scientists, and researchers aiming to harness the power of SAR imaging for various applications.
## Table of Contents

- [Introduction](#Introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Support](#support)

The Sentinel-1 data decoding capabilities of SARLens are built upon the esteemed library developed by Rich Hall, providing an expert foundation for the processing of SAR images.





## Features

**SARLens** boasts an impressive array of features, including but not limited to:

+ **Deep Learning** Model Integration: Seamless implementation of state-of-the-art deep learning models for enhanced image focusing and noise reduction.
+ **Sentinel-1** Data Support: Comprehensive support for the ingestion and processing of raw Sentinel-1 data.
+ **Range-Doppler Algorithm**: Efficient and precise focusing of SAR images through the implementation of the Range-Doppler algorithm.
+ **Parallel Processing**: Accelerated data processing utilizing parallel computing capabilities to handle large datasets.
+ **Geospatial Output**: Geocoded image output support in commonly used formats, such as GeoTIFF, for seamless integration with GIS software.
+ **Modular Design**: A modular and user-friendly design allowing for easy customization and extension of the library.

## Installation


To install SARLens, simply use the following pip command:

```
source runscripts/install.sh
```

## Usage

To utilize SARLens for decoding and focusing raw Sentinel-1 SAR images, follow these steps:

Import the necessary SARLens modules:

```
from SARLens.SARProcessor.Focus import RD
import sentinel1decoder
import sentinel1decoder.constants
import sentinel1decoder.utilities

# Create decoder
decoder = sentinel1decoder.Level0Decoder(input_file, g_level=logging.WARNING)
df = decoder.decode_metadata()
df[df["Swath Number"] == swath_number]
ephemeris = sentinel1decoder.utilities.read_subcommed_data(df)

selection = df.iloc[1000:2000] 
RangeDoppler = RD(decoder, selection, ephemeris)
logging.info(f"selected {RangeDoppler.selection.shape[0]} lines")
# Process the raw data using the Range-Doppler algorithm:
RangeDoppler.process()
print("Done")

# TODO:
focused_image = processor.range_doppler(raw_data)
Save the focused image in GeoTIFF format:
focused_image.save("path/to/output/focused_image.tif")
```

## Documentation

For comprehensive documentation of SARLens, including detailed descriptions of its methods, classes, and deep learning models, visit the official documentation website.

## Contributing

We welcome contributions to the SARLens project from the scientific and remote sensing communities. To contribute, please follow the contribution guidelines.

## License

### SARLens is released under the MIT License.

## Citation

If you employ SARLens in your research or applications, please cite our work:

## bibtex
@misc{yourusername2023sar-lens,
  title={SARLens: A Python Library for Focusing Onboard Satellite SAR Images by Deep Learning Techniques},
  author={Roberto Del Prete and Federica Biancucci},
  year={2023},
  eprint={arXiv:xxxx.xxxxx},
  url={https://github.com/sirbastiano/SARLens},
}

## Support

For any inquiries, issues, or suggestions, please visit the SARLens GitHub repository and submit an issue or pull request.

