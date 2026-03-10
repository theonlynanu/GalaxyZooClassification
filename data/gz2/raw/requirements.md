# CSV Data Requirements

The following CSV files must be in this subdirectory (`data/gz2/raw/`):

- gz2_filename_mapping.csv
  - A centralized mapping of image filenames to object ids used in labeling and metadata.
  - [Page with Download](https://zenodo.org/records/3565489)
  - [Direct Download Link](https://zenodo.org/records/3565489/files/gz2_filename_mapping.csv?download=1)
- gz2_hart16.csv
  - The primary labels, as a series of morphology vote fractions on the GZ2 decision tree. Indexed by object id.
  - [Page with Download](https://data.galaxyzoo.org/#section-8) (Table 1)
  - [Direct Download Link](https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz)
- gz2sample.csv
  - Metadata for images, including redshift, magnitude, petrosian apparent magnitude, etc.
  - [Page with Download](https://data.galaxyzoo.org/#section-8) (Bottom table labelled "SDSS metadata for GZ2")
  - [Direct Download Link](https://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/gz2sample.csv.gz)

Images can be downloaded [here](https://zenodo.org/records/3565489) (or via this [direct download link](https://zenodo.org/records/3565489/files/images_gz2.zip?download=1)), and should be unzipped and stored in `data/gz2/images`.
