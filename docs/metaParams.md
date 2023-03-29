
# Sentinel-1 Metadata Parameters

Sentinel-1 is a satellite mission developed by the European Space Agency (ESA) to provide all-weather, day-and-night imaging of Earth's surface. The mission uses Synthetic Aperture Radar (SAR) to acquire data, which is processed and analyzed to generate images and other geospatial products. The SAR data is stored in a raw format, which includes various parameters that provide information about the data and how it was acquired. These parameters include the Packet Version Number, Packet Type, Coarse Time, Fine Time, Calibration Mode, Signal Type, and others. Understanding these parameters is essential for processing and analyzing the raw data and generating useful geospatial products.

## Ancillary Information

<table>
  <tr>
    <th>Element Name</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Packet Version Number</td>
    <td>The Packet Version Number refers to the version number of the packet format. It is a 4-bit field that indicates the version of the packet protocol in use.</td>
  </tr>
  <tr>
    <td>Packet Type</td>
    <td>The Packet Type field identifies the type of data contained in the packet. This field is used to distinguish between different types of packets, such as data packets, synchronization packets, or command packets.</td>
  </tr>
  <tr>
    <td>Secondary Header Flag</td>
    <td>The Secondary Header Flag field indicates whether the packet includes a secondary header. The secondary header provides additional information about the packet and is included in packets that require additional information beyond the primary header.</td>
  </tr>
  <tr>
    <td>PID</td>
    <td>The Packet IDentifier (PID) field identifies the type of data contained in the packet. This field is used to distinguish between different types of packets, such as data packets, synchronization packets, or command packets.</td>
  </tr>
  <tr>
    <td>PCAT</td>
    <td>The Packet CATegory (PCAT) field provides additional information about the packet type. It indicates the general category of the packet, such as data packet, telemetry packet, or telecommand packet.</td>
  </tr>
  <tr>
    <td>Sequence Flags</td>
    <td>The Sequence Flags field provides information about the sequence of packets. It includes fields such as sequence count, sequence continuity flag, and sequence data length.</td>
  </tr>
  <tr>
    <td>Packet Sequence Count</td>
    <td>The Packet Sequence Count field is a counter that increments with each packet sent by the transmitter. This field is used to keep track of the packets and ensure that they are received in the correct order.</td>
  </tr>
  <tr>
    <td>Packet Data Length</td>
    <td>The Packet Data Length field indicates the length of the packet data in bytes.</td>
  </tr>
  <tr>
    <td>Coarse Time</td>
    <td>The Coarse Time field is a 32-bit field that provides the coarse time information for the data. This field is used to identify the time at which the data was acquired.</td>
  </tr>
  <tr>
    <td>Fine Time</td>
    <td>The Fine Time field is a 32-bit field that provides the fine time information for the data. This field is used to identify the time at which the data was acquired.</td>
  </tr>
  <tr>
    <td>Sync</td>
    <td>The Sync field is a 16-bit field that indicates the start of the packet. This field is used to synchronize the receiver with the transmitter.</td>
  </tr>
  <tr>
    <td>Data Take ID</td>
    <td>The Data Take ID field identifies the data take to which the packet belongs. A data take is a continuous period of data acquisition during which the instrument operates in a specific mode.</td>
  </tr>
  <tr>
    <td>ECC Number</td>
    <td>The Error Correction Code (ECC) Number field is used to identify the type of error correction code used for the data. The ECC is used to correct errors that may occur during transmission or storage of the data.</td>
      </tr>
  <tr>
    <td>Test Mode</td>
    <td>The Test Mode field indicates whether the data was acquired in test mode or in operational mode.</td>
  </tr>
  <tr>
    <td>Rx Channel ID</td>
    <td>The Receiver (Rx) Channel ID field identifies the receiver channel used to acquire the data. The Sentinel-1 satellite has two receivers, each with four channels, and this field identifies which channel was used.</td>
  </tr>
  <tr>
    <td>Instrument Configuration ID</td>
    <td>The Instrument Configuration ID field identifies the configuration of the instrument used to acquire the data.</td>
  </tr>
  <tr>
    <td>Sub-commutated Ancillary Data Word Index</td>
    <td>The Sub-commutated Ancillary Data Word Index field identifies the index of the sub-commutated ancillary data word. The sub-commutated ancillary data word provides additional information about the data, such as the position of the antenna or the configuration of the instrument.</td>
  </tr>
  <tr>
    <td>Sub-commutated Ancillary Data Word</td>
    <td>The Sub-commutated Ancillary Data Word field provides additional information about the data, such as the position of the antenna or the configuration of the instrument. The sub-commutated ancillary data word is typically updated at a lower rate than the main data and provides additional information about the data that can be used for processing or analysis.</td>
  </tr>
  <tr>
    <td>Space Packet Count</td>
    <td>The Space Packet Count field is a counter that increments with each space packet sent by the transmitter. This field is used to keep track of the packets and ensure that they are received in the correct order.</td>
  </tr>
  <tr>
    <td>PRI Count</td>
    <td>The PRI Count field is a counter that increments with each PRI (Pulse Repetition Interval) in the data. This field is used to keep track of the timing information for the data.</td>
  </tr>
  <tr>
    <td>Error Flag</td>
    <td>The Error Flag field indicates whether an error occurred during transmission or processing of the data.</td>
  </tr>
  <tr>
    <td>BAQ Mode</td>
    <td>The Block Adaptive Quantization (BAQ) Mode field identifies the mode used for quantizing the data. The BAQ mode determines how the data is compressed and is selected based on the characteristics of the data and the desired quality of the output.</td>
  </tr>
  <tr>
    <td>BAQ Block Length</td>
    <td>The BAQ Block Length field indicates the length of the BAQ block used for quantizing the data.</td>
  </tr>
  <tr>
    <td>Range Decimation</td>
    <td>The Range Decimation field indicates the level of decimation used for the data. Range decimation is the process of reducing the amount of data by selecting a subset of the data points.</td>
  </tr>
  <tr>
    <td>Rx Gain</td>
    <td>The Receiver (Rx) Gain field indicates the gain setting for the receiver used to acquire the data. The gain setting determines the sensitivity of the receiver and affects the quality of the output data.</td>
  </tr>
  <tr>
    <td>Tx Ramp Rate</td>
    <td>The Transmitter (Tx) Ramp Rate field indicates the rate at which the frequency of the transmitted signal changes over time. The ramp rate is used to provide range resolution and is selected based on the characteristics of the data and the desired quality of the output.</td>
  </tr>
  <tr>
    <td>Tx Pulse Start Frequency</td>
    <td>The Transmitter (Tx) Pulse Start Frequency field indicates the starting frequency of the transmitted pulse.</td>
  </tr>
  <tr>
    <td>Tx Pulse Length</td>
    <td>The Transmitter (Tx) Pulse Length field indicates the length of the transmitted pulse.
    </td>
  </tr>
  <tr>
    <td>Rank</td>
    <td>The Rank field indicates the rank of the data. Rank is a measure of the quality of the data and is used to determine the processing parameters that should be used to generate the output data.</td>
  </tr>
  <tr>
    <td>PRI</td>
    <td>The Pulse Repetition Interval (PRI) field indicates the time between transmitted pulses. The PRI is used to provide range resolution and is selected based on the characteristics of the data and the desired quality of the output.</td>
  </tr>
  <tr>
    <td>SWST</td>
    <td>The Synthetic Wide-Swath Time (SWST) field indicates the time at which the data was acquired for a Synthetic Aperture Radar (SAR) acquisition. The SWST is used to provide timing information for the SAR data and is used to determine the processing parameters that should be used to generate the output data.</td>
  </tr>
  <tr>
    <td>SWL</td>
    <td>The Synthetic Wide-Swath Length (SWL) field indicates the length of the data acquired for a Synthetic Aperture Radar (SAR) acquisition. The SWL is used to provide length information for the SAR data and is used to determine the processing parameters that should be used to generate the output data.</td>
  </tr>
  <tr>
    <td>SAS SSB Flag</td>
    <td>SAS stands for "SAR Auxiliary Subsystem," which is a part of the SAR processing system responsible for auxiliary data management. The auxiliary data are essential for precise SAR image processing and geolocation. SSB is an abbreviation for "Single Sideband," which refers to a technique used in radio communications to reduce the bandwidth required to transmit a signal. The "SAS SSB Flag" is a metadata attribute included in the Sentinel-1 RAW product files, indicating the status of the single sideband processing applied to the SAR data. The flag can be set to "True" or "False." If it is set to "True," it means the single sideband processing has been applied correctly. If it is set to "False," there may be issues in the SAR data processing or quality, and users should be cautious when using the data for analysis or applications. Users of Sentinel-1 RAW data can check the metadata of the product files to determine the status of the "SAS SSB Flag" and take necessary actions if the flag indicates potential issues with the data.</td>
  </tr>
  <tr>
    <td>Polarisation</td>
    <td>The Polarisation field indicates the polarization of the data. Polarization is a measure of the orientation of the electromagnetic waves used to acquire the data.</td>
  </tr>
  <tr>
    <td>Temperature Compensation</td>
    <td>The Temperature Compensation field indicates whether temperature compensation was applied to the data. Temperature compensation is used to correct for variations in the instrument caused by changes in temperature.</td>
  </tr>
  <tr>
    <td>Calibration Mode</td>
    <td>The Calibration Mode field indicates whether the data was acquired in calibration mode. Calibration mode is used to calibrate the instrument and ensure that the output data is accurate.</td>
  </tr>
  <tr>
    <td>Tx Pulse Number</td>
    <td>The Transmitter (Tx) Pulse Number field indicates the number of the transmitted pulse. This field is used to identify the transmitted pulse and is used in processing the data.</td>
  </tr>
  <tr>
    <td>Signal Type</td>
    <td>The Signal Type field indicates the type of signal used to acquire the data. The Sentinel-1 satellite uses a variety of signals, including chirp signals and pulse compression signals , to acquire different types of data.</td>
  </tr>
  <tr>
    <td>Swap Flag</td>
    <td>The Swap Flag field indicates whether the data was acquired with a swap of the in-phase and quadrature-phase components. Swapping the in-phase and quadrature-phase components can be used to correct for phase errors in the data.</td>
  </tr>
  <tr>
    <td>Swath Number</td>
    <td>The Swath Number field identifies the swath to which the data belongs. A swath is a strip of data acquired by the instrument, and the Sentinel-1 satellite can acquire data in multiple swaths.</td>
  </tr>
  <tr>
    <td>Number of Quads</td>
    <td>The Number of Quads field indicates the number of quadrature signals used to acquire the data. The Sentinel-1 satellite uses quadrature signals to acquire the data, which provides a measure of the phase and amplitude of the signal.</td>
  </tr>
</table>