# Overview

I use this repo primarily for random utility scripts I want to share that don't fit in any of the other repos where I put random utility scripts.  If for some reason you want to know more about random utility scripts I write, most of them live in the [MegaDetector](https://github.com/agentmorris/MegaDetector) repo.

I also use this repo as a staging area for projects that don't really need their own repos yet.

# Contents

Take this list with a grain of salt, I don't update it regularly.

* [drone-datasets.md](drone-datasets.md): a list of datasets with annotated aerial/drone/satellite imagery for wildlife surveys.
* [tree-ring-test.py](tree-ring-test.py): my first experiments with the [TRG-ImageProcessing](https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing/) package for detecting tree rings in tree slices.
* [deepforest-test.py](deepforest-test.py): my first experiments with the [DeepForest](https://deepforest.readthedocs.io/en/latest/) package for detecting trees in aerial imagery. 
* [gdal_convert_nan_to_nodata.py](gdal_convert_nan_to_nodata.py): script to convert all NaN values in a folder full of GeoTIFFs to a specified NODATA value, including updating the image header to specify the NODATA value (using GDAL)

# Things that "graduated" from this repo

* [Bird detector for the USGS Izembek goose survey](https://github.com/agentmorris/usgs-geese)
* [Bird detector for the UND duck survey](https://github.com/agentmorris/und-ducks)
* [noaa-fish](https://github.com/agentmorris/noaa-fish): code for preparing the <a href="https://lila.science/datasets/noaa-puget-sound-nearshore-fish">NOAA Puget Sound Nearshore Fish</a> dataset for release on on <a href="https://lila.science/">lila.science</a>, and code for training a preliminary detector on that dataset

