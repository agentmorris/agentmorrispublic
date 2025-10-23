# Datasets with annotated wildlife in drone/aerial images

Permalink to this page:

<http://lila.science/aerialdata>


## Contents

* <a href="#overview">Overview</a>
* <a href="#publicly-available-datasets">Publicly available datasets</a>
* <a href="#datasets-available-by-request">Datasets available by request</a>
* <a href="#datasets-added-recently-that-we-havent-had-a-chance-to-dig-into-yet">Datasets added recently that we haven't had a chance to dig into yet</a>
* <a href="#publicly-available-models-for-wildlife-detection-in-droneaerial-images">Publicly-available models for wildlife detection in drone/aerial images</a>
* <a href="#platformssystems-for-wildlife-detection-in-droneaerial-images">Platforms/systems for wildlife detection in drone/aerial images</a>
* <a href="#oss-repos-about-wildlife-detection-in-droneaerial-images">OSS repos about wildlife detection in drone/aerial images</a>

## Overview

This is a list of datasets with annotated aerial/drone/satellite imagery for wildlife surveys.  I've tried to collect basic standardized metadata for each dataset, and to provide sample code in a similar format for each dataset that can match annotations to images and render a sample image.  The goal of this exercise was to make it easier to assess whether existing datasets could be useful for training new models, e.g. if you want to find birds in images with a particular gestalt, we wanted to make it easier for you to get the gestalt of existing datasets, and have a starting point for parsing that data.

Everything listed here is also either listed on LILA's <a href="https://lila.science/otherdatasets">list of other conservation datasets</a> or is on <a href="https://lila.science">LILA</a>.

This list is maintained by <a href="https://dmorris.net">Dan Morris</a>, but it began life as a thread on the AI for Conservation Slack, and was initially assembled with help from <a href="https://www.linkedin.com/in/zhongqi-miao-17a50084/">Zhongqi Miao</a>, <a href="https://www.linkedin.com/in/kalindifonda">Kalindi Fonda</a>, <a href="https://www.linkedin.com/in/aakash-gupta-5ky">Aakash Gupta</a>, <a href="https://www.linkedin.com/in/jveitchmichaelis/">Josh Veitch-Michaelis</a>, and <a href="https://www.linkedin.com/in/edbayes/">Ed Bayes</a>.

Email <a href="mailto:agentmorris+dronedatasets@gmail.com">Dan</a> if anything seems off, or if you know of datasets I'm missing.

As a bonus, this page is also a temporary holding place for a list of <i><a href="#publicly-available-models-for-wildlife-detection-in-droneaerial-images">models</a></i> for wildlife detection in aerial/drone imagery.

## Publicly available datasets

### Improving the precision and accuracy of animal population estimates with aerial image object detection

Aerial images with 4305 bounding boxes on zebra, giraffe, and elephants
  
Eikelboom JA, Wind J, van de Ven E, Kenana LM, Schroder B, de Knegt HJ, van Langevelde F, Prins HH. Improving the precision and accuracy of animal population estimates with aerial image object detection. Methods in Ecology and Evolution. 2019 Nov;10(11):1875-87.

* 5.7 GB, downloadable via http from 4TU (<a href="https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1">download link</a>)
* Metadata in csv format
* Categories: elephant, zebra, giraffe
* Vehicle type: plane
* Image information: 561 RGB images
* Annotation information: 4305 boxes
* Typical animal size in pixels: 50
* License: CC0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-eikelboom-savanna.py">preview-eikelboom-savanna.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/eikelboom_savanna_sample_image_annotated.jpg" width=700>
  
  
### UAV-derived waterfowl thermal imagery dataset

8976 bounding boxes on waterfowl in UAV-derived thermal images
  
Hu Q, Smith J, Woldt W, Tang Z. UAV-derived waterfowl thermal imagery dataset. Mendeley Data, V4. 2021.

* 4.1 GB, downloadable via http from Mendeley Data (<a href="https://data.mendeley.com/datasets/46k66mz9sz/4">download link</a>)
* Metadata in csv format
* Categories: waterfowl
* Vehicle type: plane
* Image information: 541 thermal images (unannotated RGB images included as a visual reference)
* Annotation information: 8976 boxes
* Typical animal size in pixels: 7
* License: CC BY 4.0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-hu-thermal.py">preview-hu-thermal.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/hu_thermal_sample_image_annotated.jpg" width=700>
  
  
### Drones count wildlife more accurately and precisely than humans

Images of 10 life-sized, replica seabird colonies containing a known number of fake birds taken from four different heights (30m, 60m, 90m and 120m)
  
Hodgson JC, Mott R, Baylis SM, Pham TT, Wotherspoon S, Kilpatrick AD, Raja Segaran R, Reid I, Terauds A, Koh LP. Drones count wildlife more accurately and precisely than humans. Methods in Ecology and Evolution. 2018 May;9(5):1160-7.

* 50 MB, downloadable via http from Dryad (<a href="https://datadryad.org/stash/dataset/doi:10.5061/dryad.rd736">download link</a>)
* Metadata in csv format
* Categories: fake seabirds
* Vehicle type: drone
* Image information: 40 RGB images
* Annotation information: 1560 counts
* Typical animal size in pixels: variable
* License: CC0

<img src="https://raw.githubusercontent.com/KalindiFonda/bAIo/main/aerial-drone-data-preview/drone_count_seabird_colony_60m_sample.jpg" width=700>
  
  
### Counting animals in aerial images with a density map estimation model

137365 point annotations on penguins in RGB UAV images
  
Qian Y, Humphries G, Trathan P, Lowther A, Donovan C.  Counting animals in aerial images with a density map estimation model [Data set]. 2023.

* 300 MB, downloadable via http from Zenodo (<a href="https://zenodo.org/record/7702635#.ZChnoHZBxD8">download link</a>)
* Metadata in json format (LabelBox standard)
* Categories: brush-tailed penguins
* Vehicle type: plane
* Image information: 753 RGB images (orthorectified)
* Annotation information: 137365 points
* Typical animal size in pixels: 30
* License: CC0 
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-qian-penguins.py">preview-qian-penguins.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/qian_penguins_sample_image_annotated.jpg" width=700>
  
  
### Data from: A convolutional neural network for detecting sea turtles in drone imagery

1902 point annotations on sea turtles in drone images
  
Gray PC, Fleishman AB, Klein DJ, McKown MW, Bezy VS, Lohmann KJ, Johnston DW. A convolutional neural network for detecting sea turtles in drone imagery. Methods in Ecology and Evolution. 2019 Mar;10(3):345-55.

* 7.24 GB, downloadable via http from Zenodo (<a href="https://zenodo.org/record/5004596#.ZChnr3ZBxD8">download link</a>)
* Metadata in csv format
* Categories: olive ridley turtle
* Vehicle type: drone
* Image information: 1059 NIR images (false color NIR rendered to RGB)
* Annotation information: 1902 points
* Typical animal size in pixels: 10
* License: CC0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-gray-turtles.py">preview-gray-turtles.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/gray_turtles_sample_image_annotated.jpg" width=700>
  
  
### The Aerial Elephant Dataset

15581 point annotations on elephants in aerial images
  
Naude J, Joubert D. The Aerial Elephant Dataset: A New Public Benchmark for Aerial Object Detection. InProceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops 2019 (pp. 48-55).

* 16.3 GB, downloadable via http from Zenodo (<a href="https://zenodo.org/record/3234780">download link</a>)
* Metadata in csv format
* Categories: elephant
* Vehicle type: drone
* Image information: 2074 RGB images
* Annotation information: 15581 points
* Typical animal size in pixels: 75
* License: CC0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-aerial-elephants.py">preview-aerial-elephants.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/aerial_elephants_sample_image_annotated.jpg" width=700>
  
  
### A global model of bird detection in high resolution airborne images using computer vision

386638 box annotations on 23765 drone images from 13 ecosystems
  
Weinstein BG, Garner L, Saccomanno VR, Steinkraus A, Ortega A, Brush K, Yenni G, McKellar AE, Converse R, Lippitt CD, Wegmann A. A general deep learning model for bird detection in high-resolution airborne imagery. Ecological Applications. 2022 Dec:e2694.

* 29 GB, downloadable via http from Zenodo (<a href="https://zenodo.org/record/5033174">download link</a>)
* Metadata in csv format
* Categories: bird
* Vehicle type: variable
* Image information: 23765 RGB images
* Annotation information: 386638 boxes
* Typical animal size in pixels: 35
* License: CC BY 4.0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-weinstein-birds.py">preview-weinstein-birds.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/weinstein_birds_sample_image_annotated.jpg" width=700>
  
  
### Aerial Photo Imagery from Fall Waterfowl Surveys, Izembek Lagoon, Alaska, 2017-2019

631349 point annotations on waterfowl on 110,067 aerial images
  
Weiser EL, Flint PL, Marks DK, Shults BS, Wilson HM, Thompson SJ, Fischer JB. Aerial photo imagery from fall waterfowl surveys, Izembek Lagoon, Alaska, 2017-2019: U.S. Geological Survey data release. 2022.

* 1.82 TB, downloadable via http from USGS (<a href="https://alaska.usgs.gov/products/data.php?dataid=484">download link</a>)
* A slightly-more-curated but technically-incomplete version of this dataset (many of the blank images hve been removed) is [hosted on LILA](https://lila.science/datasets/izembek-lagoon-waterfowl/).
* Metadata in csv, json format (CountThings standard)
* Categories: brant goose, emperor goose, canada goose, gull, other
* Vehicle type: plane
* Image information: 110667 RGB images
* Annotation information: 631349 points
* Typical animal size in pixels: 50
* License: unspecified, but public domain implied (USGS source)
* Code to render sample annotated image from the original dataset: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-weiser-waterfowl.py">preview-weiser-waterfowl.py</a>
* Code to render sample annotated image from the LILA dataset: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-weiser-waterfowl-lila.py">preview-weiser-waterfowl-lila.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/weiser_waterfowl_sample_image_annotated.jpg" width=700>
  
  
### Data from: Drones and deep learning produce accurate and efficient monitoring of large-scale seabird colonies

44966 bounding boxes on drone images of black-browed albatrosses and southern rockhopper penguins
  
Hayes MC, Gray PC, Harris G, Sedgwick WC, Crawford VD, Chazal N, Crofts S, Johnston DW. Data from: Drones and deep learning produce accurate and efficient monitoring of large-scale seabird colonies. Duke Research Repository. doi. 2020;10:r4dn45v9g.

* 20.5 GB, downloadable via Globus from Globus (<a href="https://research.repository.duke.edu/concern/datasets/kp78gh20s?locale=en">download link</a>)
* Metadata in csv format
* Categories: black-browed albatross, southern rockhopper penguin
* Vehicle type: drone
* Image information: 3947 RGB images
* Annotation information: 44966 boxes
* Typical animal size in pixels: 300
* License: CC0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-hayes-seabirds.py">preview-hayes-seabirds.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/hayes_seabirds_sample_image_annotated.jpg" width=700>
  
  
### Cattle detection and counting in UAV images based on convolutional neural networks

1919 bounding boxes on cattle in drone images, with individual IDs
  
Shao W, Kawakami R, Yoshihashi R, You S, Kawase H, Naemura T. Cattle detection and counting in UAV images based on convolutional neural networks. International Journal of Remote Sensing. 2020 Jan 2;41(1):31-52.

* 3.2 GB, downloadable via http from University of Tokyo (<a href="http://bird.nae-lab.org/cattle/">download link</a>)
* Metadata in txt format
* Categories: cattle
* Vehicle type: drone
* Image information: 663 RGB images
* Annotation information: 1919 boxes
* Typical animal size in pixels: 90
* License: unspecified
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-shao-cattle.py">preview-shao-cattle.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/shao_cattle_sample_image_annotated.jpg" width=700>
  
  
### NOAA Fisheries Steller Sea Lion Population Count

948 aerial images of sea lions with counts for each image
  
* 103 GB, downloadable via http or torrent from Kaggle (<a href="https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count">download link</a>)
* Metadata in csv format
* Categories: steller sea lion
* Vehicle type: plane
* Image information: 948 RGB images
* Annotation information: 948 counts
* Typical animal size in pixels: 75
* License: unspecified (public domain implied, NOAA source)
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-steller-sea-lion-count.py">preview-steller-sea-lion-count.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/steller_sea_lion_count_sample_image.jpg" width=700>
  
  
### Right Whale Recognition

4544 images of right whales with individual IDs
  
* 10 GB, downloadable via http from Kaggle (<a href="https://www.kaggle.com/c/noaa-right-whale-recognition">download link</a>)
* Metadata in csv format
* Categories: right whale
* Vehicle type: helicopter
* Image information: 11468 RGB images
* Annotation information: 4544 individual IDs
* Typical animal size in pixels: 1500
* License: unspecified (public domain implied, NOAA source)
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-right-whale-recognition.py">preview-right-whale-recognition.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/right_whale_recognition_sample_image.jpg" width=700>
  
  
### NOAA Arctic Seals 2019

Around 14000 bounding boxes on seals in 44185 color/thermal image pairs 
  
Alaska Fisheries Science Center, 2021: A Dataset for Machine Learning Algorithm Development.

* 1 TB, downloadable via azcopy from LILA (<a href="https://lila.science/datasets/noaa-arctic-seals-2019/">download link</a>)
* Metadata in csv format
* Categories: ringed_seal, ringed_pup, unknown_seal, bearded_pup, bearded_seal, unknown_pup
* Vehicle type: plane
* Image information: 44185 RGB+IR images (rGB and IR are different resolutions, but registered well, and annotations for every animal are provided for both images)
* Annotation information: 14311 boxes
* Typical animal size in pixels: 55
* License: CDLA-permissive
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-noaa-arctic-seals.py">preview-noaa-arctic-seals.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/noaa_arctic_seals_sample_image_annotated.jpg" width=700>
  
  
### Aerial Seabirds West Africa

High-resolution aerial RGB imagery obtained from a census of breeding seabirds in West Africa in 2019, with 21516 point annotations on seabirds
  
Kellenberger B, Veen T, Folmer E, Tuia D. 21,000 birds in 4.5 h: efficient large-scale seabird detection with machine learning. Remote Sensing in Ecology and Conservation. 2021.

* 2.2 GB, downloadable via http or azcopy from LILA (<a href="https://lila.science/datasets/aerial-seabirds-west-africa/">download link</a>)
* Metadata in csv format
* Categories: great white pelican, royal tern, caspian tern, slender-billed gull, gray-headed gull, great cormorant
* Vehicle type: plane
* Image information: single aerial orthomosaic RGB images
* Annotation information: 21516 points
* Typical animal size in pixels: 30
* License: CDLA-permissive
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-aerial-seabirds-west-africa.py">preview-aerial-seabirds-west-africa.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/aerial_seabirds_west_africa_sample_image_annotated.jpg" width=700>
  
  
### Conservation Drones

Thermal drone videos with 166221 boxes and object IDs on humans, elephants, and several other animals
  
Bondi E, Jain R, Aggrawal P, Anand S, Hannaford R, Kapoor A, Piavis J, Shah S, Joppa L, Dilkina B, Tambe M. BIRDSAI: A Dataset for Detection and Tracking in Aerial Thermal Infrared Videos.

* 3.7 GB, downloadable via http or azcopy from LILA (<a href="https://lila.science/datasets/conservationdrones">download link</a>)
* Metadata in csv format (MOT standard)
* Categories: human, elephant, giraffe, lion, dog
* Vehicle type: drone
* Image information: 61994 frames from 48 videos thermal images
* Annotation information: 166221 boxes
* Typical animal size in pixels: 35
* License: CDLA-permissive
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-conservation-drones.py">preview-conservation-drones.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/conservation_drones_sample_image_annotated.jpg" width=700>


### DAZZLE: Drone-Acquired Zebra Data for Large-scale Ecology Research

Oblique aerial videos of zebras with 162931 bounding boxes and behavioral labels (standing, grazing, etc.).

Price E, Khandelwal PC, Rubenstein DI, Ahmad A. A Framework for Fast, Large-scale, Semi-Automatic Inference of Animal Behavior from Monocular Videos. bioRxiv. 2023:2023-07.

* 96GB, downloadable via http from Keeper (<a href="https://keeper.mpdl.mpg.de/d/a9822e000aff4b5391e1/">download link</a>)
* More information available on the [dataset home page](https://www.aamirahmad.de/datasets/dazzle/)
* Metadata in json format (Labelme standard)
* Categories: zebra, person, vehicle
* Vehicle type: plane
* Image information: 30,6026 video frames (~4k resolution)
* Annotation information: 162931 boxes (4387 fully manual, 158544 semi-automated) with behavior labels
* Typical animal size in pixels: 134
* License: unspecified
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-price-zebras.py">preview-price-zebras.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/price_zebras_sample_image_annotated.jpg" width=700>


### Quantifying the movement, behaviour and environmental context of group-living animals using drones and computer vision

Drone images of ungulates and geladas with 40532 bounding boxes.

Koger B, Deshpande A, Kerby JT, Graving JM, Costelloe BR, Couzin ID. [Quantifying the movement, behaviour and environmental context of group-living animals using drones and computer vision](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/1365-2656.13904). Journal of Animal Ecology. 2023 Mar 21.

* 65GB, downloadable via http from Edmond (<a href="https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.EMRZGH">download link</a>)
* Metadata in json format (COCO standard)
* Categories: zebra, gazelle, waterbuck, buffalo, other, gelada, human
* Vehicle type: drone
* Image information: 1982 drone images
* Annotation information: 40532 boxes
* Typical animal size in pixels: 56
* License: CC0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-koger-drones.py">preview-koger-drones.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/koger_drones_sample_image_annotated.jpg" width=700>


### Data from "Deep object detection for waterbird monitoring using aerial imagery"

Kabra K, Xiong A, Li W, Luo M, Lu W, Yu T, Yu J, Singh D, Garcia R, Tang M, Arnold H. [Deep object detection for waterbird monitoring using aerial imagery](https://ieeexplore.ieee.org/document/10069986). In 2022 21st IEEE International Conference on Machine Learning and Applications (ICMLA) 2022 Dec 12 (pp. 455-460). IEEE.

* 3.7TB, downloadable from Google Drive (<a href="https://drive.google.com/file/d/1hoP1ev8Npj5m0MZWZU7LpjU9c8JYYoFe/view?usp=share_link">download link</a>)
* Metadata in .csv format
* Categories: <a href="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/kabra_birds_categories.txt">list of categories</a>
* Vehicle type: drone
* Image information: 200 drone images
* Annotation information: 23078 boxes, species and age (juvenile/adult) information for each box
* Typical animal size in pixels: 88.9
* License: unspecified
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-kabra-birds.py">preview-kabra-birds.py</a>


<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/kabra_birds_sample_image_annotated.jpg" width=700>


### KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition from Drone Videos

10 hours of UAV video from Kenyan savanna, with behavior labels.

Kholiavchenko M, Kline J, Ramirez M, Stevens S, Sheets A, Babu R, Banerji N, Campolongo E, Thompson M, Van Tiel N, Miliko J. KABR: In-Situ Dataset for Kenyan Animal Behavior Recognition From Drone Videos. Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2024.

* 54GB, downloadable via http from Google Drive (<a href="https://dirtmaxim.github.io/kabr">download link</a>)  (<a href="https://huggingface.co/datasets/imageomics/KABR">Hugging Face link</a>)
* Metadata in space-delimited table format
* Categories: (walk, graze, browse, head up, auto-groom, trot, run, occluded) for (giraffe, zebra)
* Vehicle type: drone
* Image information: 130366 videos each following a single individual, provided as ~1.1M JPGs
* Annotation information: behavior labels for individual frames
* Typical animal size in pixels: 100
* License: unspecified
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-kabr-behavior.py">preview-kabr-behavior.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/kabr_sample_image_annotated.jpg">


### MMLA-OPC

~29k frames with boxes on zebras in UAV images, collected from Ol Pejeta Conservancy in Kenya.

Kline J, Nguyen Ngoc D, Duncan H, Rondaeu Saint-Jean C, Maalouf G, Juma B, Kilwaya A, Vuyiya B, Irungu M, Njoroge W, Mutisya S, Guerin D, Costelloe B, Pastucha E, Hermansen J, Kjeld J, Watson M, Richardson T, Schultz Lundquist UP.  MMLA Ol Pejeta Conservancy (OPC) Dataset, 2025.

* 128GB, downloadable from [Hugging Face](https://huggingface.co/datasets/imageomics/mmla_opc)
* Metadata in YOLO format
* Categories: zebra
* Vehicle type: drone
* Image information: 29,268 RGB frames
* Annotation information: ~163k bounding boxes
* Typical animal size in pixels
* License: [CC 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-mmla-opc.py">preview-mmla-opc.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/mmla_opc_sample_image_annotated.jpg">


### MMLA-Wilds

~8k frames with boxes on onagers (~45k), giraffe (~7k), zebra (~2.5k) and wild dogs (14), collected at The Wilds Conservation Center in Ohio.

Kline J, Zhong A, Yablok J.   MMLA The Wilds Dataset, 2025.

* 21GB, downloadable from [Hugging Face](https://huggingface.co/datasets/imageomics/mmla_wilds)
* Metadata in YOLO format
* Categories: zebra, giraffe, onager, dog
* Vehicle type: drone
* Image information: 8,009 RGB frames
* Annotation information: bounding boxes
* Typical animal size in pixels: 1000px
* License: [CC 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-mmla-wilds.py">preview-mmla-wilds.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/mmla_wilds_sample_image_annotated.jpg">


### MMLA-Mpala

~130k frames with boxes on wild giraffe and zebra, collected at Mpala Research Center in Kenya.

Kline J, Kholiavchenko M, Zhong Alison, Ramirez M, Stevens S, Van Tiel N, Campolongo E, Thompson M, Ramesh Babu R, Banerji N, Sheets A, Magersupp M, Balasubramaniam S, Duporge I, Miliko J, Rosser N, Stewart CV, Berger-Wolf T, Rubenstein DI.

MMLA Mpala Dataset, 2025.

* 490GB, downloadable from [Hugging Face](https://huggingface.co/datasets/imageomics/mmla_mpala)
* Metadata in YOLO format
* Categories: zebra, giraffe, onager, dog
* Vehicle type: drone
* Image information: 130,102 images
* Annotation information: bounding boxes
* Typical animal size in pixels:
* License: [CC 1.0](https://creativecommons.org/publicdomain/zero/1.0/)
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-mmla-mpala.py">preview-mmla-mpala.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/mmla_mpala_sample_image_annotated.jpg">


### WAID: Wildlife Aerial Images from Drone


Mou C, Liu T, Zhu C, Cui X. Waid: A large-scale dataset for wildlife detection with drones. Applied Sciences. 2023 Sep 17;13(18):10397.IEEE/CVF Winter Conference on Applications of Computer Vision 2024.

* 1.5GB, downloadable from GitHub (as in, literally from GitHub) (<a href="https://github.com/xiaohuicui/WAID/tree/main/WAID">download link</a>)
* Metadata in YOLO format
* Categories: sheep, cattle, seal, camel, kiang, zebra (some images also include un-annotated birds)
* Vehicle type: drone
* Image information: 14,366 images, typically 640x640
* Annotation information: boxes
* Typical animal size in pixels: 166
* License: unspecified
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-waid-drones.py">preview-waid-drones.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/waid_sample_image_annotated.jpg">


### UAS Imagery of Migratory Waterfowl at New Mexico Wildlife Refuges

Converse RC, Lippitt CD, Sesnie SE, Harris GM, Butler MG, Stewart DR. Observer variability in manual-visual interpretation of UAS imagery of wildlife, with insights for deep learning applications.  In review.

* 322MB, downloadable from LILA (<a href="https://lila.science/datasets/uas-imagery-of-migratory-waterfowl-at-new-mexico-wildlife-refuges">download link</a>)
* Metadata in COCO .json format
* Categories: canada goose, sandhill crane, mallard, northern pintail, american wigeon, teal, gadwall, northern shoveler, other
* Vehicle type: drone
* Image information: 12 images if ~5kx4k, 356 images of 684x521
* Annotation information: 2243 consensus boxes
* Typical animal size in pixels: 50
* License: CC-BY-NC 4.0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-nm-waterfowl.py">preview-nm-waterfowl.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/nm_waterfowl_sample_image_annotated.png">


### Multispecies detection and identification of African mammals

Delplanque A, Foucher S, Lejeune P, Linchant J, Théau J. Multispecies detection and identification of African mammals in aerial imagery using convolutional neural networks. Remote Sensing in Ecology and Conservation. 2022 Apr;8(2):166-79.

* 12GB, downloadable from Liege University Dataverse (<a href="https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0">download link</a>)
* Metadata in COCO .json format
* Categories: alcelaphinae, buffalo, kob, warthog, waterbuck, elephant
* Vehicle type: plane
* Image information: 1297 images, each 6000x4000
* Annotation information: 10,239 boxes
* Typical animal size in pixels: 47
* License: CC-BY-NC-SA 4.0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-delplanque-mammals.py">preview-delplanque-mammals.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/delplanque_mammals_sample_image_annotated.jpg" width=700>


### SAVMAP (UAV images of Namibian wildlife)

Reinhard F, Parkan M, Produit T, Betschart S, Bacchilega B, Hauptfleisch ML, Meier P, Joost S, Tuia D. Near real-time ultrahigh-resolution imaging from unmanned aerial vehicles for sustainable land use management and biodiversity conservation in semi-arid savanna under regional and global change (SAVMAP). Zenodo.

* 3GB, downloadable from <a href="https://zenodo.org/records/1204408">Zenodo</a> or <a href="https://huggingface.co/datasets/fadel841/savmap">Hugging Face</a>
* Metadata in .geojson format
* Categories: animal
* Vehicle type: UAV
* Image information: 659 images, each 4000x3000
* Annotation information: ~7.5k polygons, though those really approximate boxes, and there are a smaller number of unique annotations. But O(thousands).
* Typical animal size in pixels: <100
* License: AFL-3.0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/aerial-drone-data-preview/preview-savmap.py">preview-savmap.py</a>

<img src="https://raw.githubusercontent.com/agentmorris/agentmorrispublic/main/aerial-drone-data-preview/savmap_thumb.jpg" width=700>


## Datasets available by request

### Identification of free-ranging mugger crocodiles by applying deep learning methods on UAV imagery

88,000 images focusing on the mugger’s dorsal body. The data was collected from 143 individuals across 19 different locations along the western part of India.
  
* 1.5 GB, downloadable via http from Dryad (<a href="https://datadryad.org/stash/landing/show?id=doi%3A10.5061%2Fdryad.s4mw6m98n">download link</a>)
* Categories: mugger crocodile
* Vehicle type: drone
* Image information: 88000 RGB images
* Annotation information:  individual ID
* Typical animal size in pixels: 1000
* License: CC0

<img src="https://raw.githubusercontent.com/KalindiFonda/bAIo/main/aerial-drone-data-preview/mugger_crocodiles_31_2_sample.jpg" width=700>


### Whales from Space

633 satellite image chips with boxes on whales
  
Cubaynes HC, Fretwell PT. Whales from space dataset, an annotated satellite image dataset of whales for training machine learning models. Scientific Data. 2022 May 27;9(1):245.

* 10 MB, downloadable via http from by request (<a href="https://www.nature.com/articles/s41597-022-01377-4#Sec8">download link</a>)
* Metadata in csv, shapefile format
* Categories: southern right whale, humpback whale, fin whale, grey whale
* Vehicle type: satellite
* Image information: 633 RGB images (150x150 chips)
* Annotation information:  boxes
* Typical animal size in pixels: 50
* License: variable
    
  
## Publicly-available models for wildlife detection in drone/aerial images

This section lists ML models one can download and run locally on drone/aerial images of wildlife (or use in cloud-based systems).  This section does not include models that exist in online platforms but can't be downloaded locally.

My hope is that this section can grow into a more structured database of models with sample code... if you want to help with that, <a href="mailto:agentmorris+dronesurvey@gmail.com">email me</a>.

I am making a very loose effort to include last-updated dates for each of these.  Those dates are meant to very loosely capture repo activity, so that if you go looking for a detector for ecosystem X, you can start with more active sources.  But I'm not digging that deep; if someone trained a detector in 2016 that is totally obsolete, but they corrected a bunch of typos in their repo in 2023, they will successfully trick my algorithm for determining the last-updated date.

When possible, the first link for each line item should get you pretty close to the model weights.

* [HerdNet](https://github.com/Alexandre-Delplanque/HerdNet) (2022, [code](https://github.com/Alexandre-Delplanque/HerdNet), [data](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5)) (custom detector for African mammals in aerial imagery)
* [Global model of bird detection](https://github.com/weecology/BirdDetector/releases) (2021, [code](https://github.com/weecology/BirdDetector), [data](https://zenodo.org/records/5033174), [paper](https://www.biorxiv.org/content/10.1101/2021.08.05.455311v2.full)) (RetinaNet on ResNet50 in PyTorch) (downloadable directly, but recommended use is via the [DeepForest package](https://deepforest.readthedocs.io/en/latest/prebuilt.html#bird-detection))
* [DeepForest's Livestock Detection Model](https://deepforest.readthedocs.io/en/v1.4.1/user_guide/02_prebuilt.html#livestock-detectors-model) (2024) (single-class detector for cows, sheep, and other large mammals in agricultural settings)
* [Izembek goose detector](https://github.com/agentmorris/usgs-geese/releases) (2023, [code](https://github.com/agentmorris/usgs-geese), [data](https://lila.science/datasets/izembek-lagoon-waterfowl)) (YOLOv5, detects birds in Izembek Lagoon in Alaska, particularly brant geese, in aerial imagery)
* [Esri Tern Detector](https://www.arcgis.com/home/item.html?id=4019a53c914947aea9621ba226ec8861) (2025, [data](https://lila.science/datasets/aerial-seabirds-west-africa)) (Mask-RCNN, trained w/ArcGIS Python API, distributed as an Esri dlpk file (which is a zipped .pth file))
* [Esri Arctic Seal Detector](https://www.arcgis.com/home/item.html?id=bb05ab8f3b7c4ec79eca613c9273ef6f) (2025, [data](https://lila.science/datasets/noaa-arctic-seals-2019)) (Faster-RCNN, trained w/ArcGIS Python API, distributed as an Esri dlpk file (which is a zipped .pth file))
* [Esri Elephant Detector](https://www.arcgis.com/home/item.html?id=4976292298c440e686aa339e52da2dbb) (2025, [data](https://zenodo.org/records/3234780)) (Faster-RCNN, trained w/ArcGIS Python API, distributed as an Esri dlpk file (which is a zipped .pth file))
* [Esri human detector](https://www.arcgis.com/home/item.html?id=42bfd5392d834c83aa21193450888a9e) (2025) (Faster-RCNN, trained w/ArcGIS Python API, distributed as an Esri dlpk file (which is a zipped .pth file))

## Platforms/systems for wildlife detection in drone/aerial images

### Platforms that are specifically related to wildlife

#### Scout

"Scout is an open hardware and open source software solution designed by Wild Me to support the analysis of large volumes of data obtained from aerial surveys of wildlife."

* [home](https://www.wildme.org/scout.html)
* [code](https://github.com/WildMeOrg/scout)
* [docs](https://docs.wildme.org/product-docs/en/scout/)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="media/scout-screenshot.png" width="600">

#### SurveyScope

"SurveyScope is a powerful web application that leverages the latest artificial intelligence (AI) to assist in the annotation of aerial-census data."

* [home](https://wildeyeconservation.org/surveyscope/)
* [code](https://github.com/WildEyeConservation/Detweb/)

#### WildAI

* [home](https://www.wildai.eu/)

Not a ton of information as of 2025.09.12, just entering beta.  Web page says "Simply upload your aerial images to our platform, where our powerful AI model automatically processes and analyzes the data for you, and then seamlessly review and verify the results with complete confidence and ease."

#### WISDAM

"WISDAM is a free, downloadable application designed for researchers (or community groups) conducting wildlife imagery surveys from either piloted aircraft or drones."

* [home](https://www.wisdamapp.org/about/)

#### AIDE

"AIDE is two things in one: a tool for manually annotating images and a tool for training and running machine (deep) learning models. Those two things are coupled in an active learning loop: the human annotates a few images, the system trains a model, that model is used to make predictions and to select more images for the human to annotate, etc."

* [home](https://github.com/microsoft/aerial_wildlife_detection)
* [code](https://github.com/microsoft/aerial_wildlife_detection) (the same link as "home", but bulleted lists with only one item are unsatisfying)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="media/aide-screenshot.jpg" width="600">

### Platforms that aren't specifically related to wildlife, but that people use for wildlife stuff

#### Label Studio

Label Studio (LS) is a general-purpose platform for data annotation (not just images, all kinds of data).  Broadly, it comes in [two flavors](https://labelstud.io/guide/label_studio_compare): LS Enterprise is a hosted (not free) service with lots of features to manage your annotation workforce; LS Community is a containerized version with largely the same front-end, but fewer annotator management and ML features.  Both are flexible and templated, to the point of maybe being a little too complicated to use for wildlife survey applications out of the box, but if this community rallies around specific templates and workflows, it may be The Right Thing.

* [home](https://labelstud.io/)
* [code](https://github.com/HumanSignal/label-studio)
* [blog post](https://github.com/weecology/LabelStudio_BlogPost/blob/main/blogpost.ipynb) from [Ben Weinstein](https://benweinstein.weebly.com/) about using using LS for wildlife annotation

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="media/labelstudio-screenshot.png" width="600">

#### BisQue

BisQue stands for "Bio-Image Semantic Query User Environment", and its mostly a cloud-based collaborative annotation platform for microscopy and biomedical images, so you might wonder why it's included here... at the end of the day, it's a platform for image annotation, with support for georeferenced images, and at least two people have mentioned using it for wildlife images, so, it counts.

* [home](https://bioimage.ucsb.edu/bisque)
* [code](https://github.com/UCSB-VRL/bisqueUCSB)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="media/bisque-screenshot.png" width="600">

#### Zooniverse

Not directly machine-learning-related, but it seems relevant in the sense that it's a good way to collect training data, and lots of the same folks who might use ML-accelerated annotation are likely to also leverage citizen scientists.

* [home](https://zooniverse.org/)
* [code](https://github.com/zooniverse)
* ["Aerial Wildlife Surveys in Africa" project](https://www.zooniverse.org/projects/simbamangu/aerial-wildlife-surveys-in-africa/)
* ["Penguins from Above" project](https://www.zooniverse.org/projects/tawakitom/penguins-from-above)
* ["Drones for Ducks" project](https://www.zooniverse.org/projects/rowan-aspire/drones-for-ducks)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="media/zooniverse-screenshot.png" width="600">

## OSS repos about wildlife detection in drone/aerial images

* POLO (modified YOLOv8 that trains on point labels)<br/>[github.com/gigumay/POLO](https://github.com/gigumay/POLO)
* DeepForest (tools for object detection in aerial images, esp trees and birds)<br/>[github.com/weecology/DeepForest](https://github.com/weecology/DeepForest)
* AIDE (interactive platform for labeling and training aerial wildlife image ML)<br/>[github.com/microsoft/aerial_wildlife_detection](https://github.com/microsoft/aerial_wildlife_detection)
* HerdNet (training and inference for ungulate detection in aerial images)<br/>[github.com/Alexandre-Delplanque/HerdNet](https://github.com/Alexandre-Delplanque/HerdNet)
* UAV Thermal Wildlife Detection (training detection models on the BIRDSAI dataset)<br/>[github.com/tiffanyyk/UAV-Thermal-IR-Wildlife-Object-Detection](https://github.com/tiffanyyk/UAV-Thermal-IR-Wildlife-Object-Detection)
* HealthyCountryAI (training and inference for UAV wildlife detection in Australia)<br/>[github.com/microsoft/HealthyCountryAI](https://github.com/microsoft/HealthyCountryAI)
* Scout (AI tools for aerial wildlife detection)<br/>[github.com/WildMeOrg/scout](https://github.com/WildMeOrg/scout)
* Global model of bird detection<br/>[github.com/weecology/BirdDetector](https://github.com/weecology/BirdDetector)
* Izembek goose detector<br/>[github.com/agentmorris/usgs-geese](https://github.com/agentmorris/usgs-geese)
* kabr-tools (behavioral analysis from drone videos)<br/>[github.com/Imageomics/kabr-tools](https://github.com/Imageomics/kabr-tools)


### OSS repos about drone/aerial wildlife datasets

* WAID (Wildlife Aerial Images from Drone)<br/>[github.com/xiaohuicui/WAID](https://github.com/xiaohuicui/WAID)


