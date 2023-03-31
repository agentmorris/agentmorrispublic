# Overview

This is an informal list of datasets with annotated aerial/drone/satellite imagery for wildlife surveys.  Everything listed here is also either listed on LILA's <a href="https://lila.science/otherdatasets">list of other conservation datasets</a> or is on <a href="https://lila.science">LILA</a>.

This is also in a very temporary state, while I sort out what's what, whether this is redundant with someone else's list, etc.  So... YMMV.  Check back in a few weeks for what will hopefully be either a cleaned-up list or a link to a cleaned-up list.

# Data sets

* Improving the precision and accuracy of animal population estimates with aerial image object detection  
https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1

* UAV-derived waterfowl thermal imagery dataset  
https://data.mendeley.com/datasets/46k66mz9sz/4

* Drones count wildlife more accurately and precisely than humans  
https://datadryad.org/stash/dataset/doi:10.5061/dryad.rd736

* Identification of free-ranging mugger crocodiles by applying deep learning methods on UAV imagery  
https://datadryad.org/stash/landing/show?id=doi%3A10.5061%2Fdryad.s4mw6m98n

* Counting animals in aerial images with a density map estimation model  
https://zenodo.org/record/7702635

* Data from: A convolutional neural network for detecting sea turtles in drone imagery  
https://zenodo.org/record/5004596

* The Aerial Elephant Dataset  
https://zenodo.org/record/3234780

* A global model of bird detection...  
https://zenodo.org/record/5033174

* Aerial Photo Imagery from Fall Waterfowl Surveys, Izembek Lagoon, Alaska, 2017-2019  
https://alaska.usgs.gov/products/data.php?dataid=484

* Data from: Drones and deep learning produce accurate and efficient monitoring of large-scale seabird colonies  
https://research.repository.duke.edu/concern/datasets/kp78gh20s?locale=en

* Cattle detection and counting in UAV images based on convolutional neural networks  
http://bird.nae-lab.org/cattle/

* Whales from Space  
https://ramadda.data.bas.ac.uk/repository/entry/show?entryid=c1afe32c-493c-4dc7-af9f-649593b97b2c

* NOAA Fisheries Steller Sea Lion Population Count  
https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count

* Right Whale Recognition  
https://www.kaggle.com/c/noaa-right-whale-recognition

* NOAA Arctic Seals 2019  
https://lila.science/datasets/noaa-arctic-seals-2019/

* Aerial Seabirds West Africa  
https://lila.science/datasets/aerial-seabirds-west-africa/

* Conservation Drones  
https://lila.science/datasets/conservationdrones

# Metadata I'd like to collect for each dataset

* Name (included above)
* URL (included above)
* Short description
* Citation
* Approximate size in GB
* Download mechanism (http direct download, Globus, etc.)
* Metadata raw format (.json, .csv, etc.)
* Metadata standard, if applicable (e.g. COCO, YOLO)
* Species present (proper taxonomic names not necessary, just common names... e.g. "albatross" or "elephant")
* Vehicle type: drone/aerial/satellite
* Channels: RGB, RGB+IR, thermal, etc.
* Ground resolution (i.e., pixel resolution in m)
* Approximate typical size of an animal in pixels (double-check this, but it should be roughly consistent with "rough animal size in meters divided by pixel size in meters")
* Sample image or image patch, ideally with annotations displayed for that image (this requires us to make sure annotations can actually be aligned to images).  Something like [this](http://lila.science/wp-content/uploads/2021/04/noaa_seals_2019_web.png), potentially useful as a thumbnail.


