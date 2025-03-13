# Datasets with annotated fish in marine/freshwater imagery/video

## Also see

* [Filippo Varini](https://www.linkedin.com/in/filippo-varini/?originalSubdomain=uk) has [picked up where this page leaves off](https://github.com/filippovarini/fish-datasets); if you are looking for a comprehensive list of fish datasets with sample code, start there.  Now that Filippo's list exists, the page you're looking at now is almost obsolete, other than the <a href="#publicly-available-models-for-fish-detection">list of models</a>.
* Though there's almost no detail there, I still try to keep [LILA's list of marine image datasets](<https://lila.science/otherdatasets#images-marine-fish>
) up to date.

## TOC

* <a href="#overview">Overview</a>
* <a href="#publicly-available-datasets">Publicly available datasets</a>
* <a href="#publicly-available-models-for-fish-detection">Publicly-available models for fish detection</a>

## Overview

This is (theoretically) a list of datasets with annotated marine/freshwater imagery, suitable for training fish detectors/classifiers.  Right now, it's not exactly that, it's just a link to a list of datasets.  But I'd like it to grow into something analogous to the (list of datasets with annotated wildlife in drone/aerial images)[drone-datasets.md], with standardized metadata for each dataset, and consistent sample code for match annotations to images and rendering sample images.

I haven't actually done that part yet, so for now, the "datasets" portion of this page is just a link to a less-structured place where I'm keeping a list of datasets.  The <i>models</i> list at the bottom of the page is slightly more comprehensive.

## Publicly available datasets

As per above, this section is currently a placeholder for a hypothetical future where someone collects standardized metadata and sample code for each dataset.  For now, I'm going to do this for just <i>one</i> dataset, to paint the picture.  But the real list of relevant datasets is at:

<https://lila.science/otherdatasets#images-marine-fish>

Here's the one dataset, to provide an example of what I'd like to do for all the other datasets...

### NOAA Puget Sound Nearshore Fish 2017-2018

Images with 67990 bounding boxes on fish and crustaceans
  
Farrell DM, Ferriss B, Sanderson B, Veggerby K, Robinson L, Trivedi A, Pathak S, Muppalla S, Wang J, Morris D, Dodhia R. A labeled data set of underwater images of fish and crab species from five mesohabitats in Puget Sound WA USA. Scientific Data. 2023 Nov 13;10(1):799.

* 7 GB, downloadable via http or gsutil from LILA (<a href="https://lila.science/datasets/noaa-puget-sound-nearshore-fish">download link</a>)
* Metadata in COCO format
* Categories: fish, crustacean
* Image information: 77,739 images from shellfish aquaculture farms in the Northeast Pacific
* Annotation information: 67990 boxes
* License: CDLA-permissive 1.0
* Code to render sample annotated image: <a href="https://github.com/agentmorris/agentmorrispublic/blob/main/fish-data-preview/preview-noaa-psnf.py">preview-noaa-psnf.py</a>

<img src="http://lila.science/wp-content/uploads/2022/07/noaa-estuary-thumb-800.png" width=700>


## Publicly-available models for fish detection

This section lists ML models one can download and run locally on images/video of fish (or use in cloud-based systems).  This section does not include models that exist in online platforms but can't be downloaded locally.

My hope is that this section can grow into a more structured database of models with sample code... if you want to help with that, <a href="mailto:agentmorris+fishsurvey@gmail.com">email me</a>.

I am making a very loose effort to include last-updated dates for each of these.  Those dates are meant to very loosely capture repo activity, so that if you go looking for a detector for ecosystem X, you can start with more active sources.  But I'm not digging that deep; if someone trained a detector in 2016 that is totally obsolete, but they corrected a bunch of typos in their repo in 2023, they will successfully trick my algorithm for determining the last-updated date.

When possible, the first link for each line item should get you pretty close to the model weights.

* [SharkTrack](https://github.com/filippovarini/sharktrack) (YOLOv8n detector and multi-frame tracker for sharks) ([home](https://www.fvarini.com/sharktrack))
* [AI for the Ocean Fish and Squid Detector](https://zenodo.org/records/7430331) (YOLOv5s, trained on 5600 images in the Eastern Pacific) (2022, [code](https://github.com/heinsense2/AIO_CaseStudy))
* [MegaFishDetector](https://github.com/warplab/megafishdetector/blob/main/MODEL_ZOO.md) (YOLOv5 models (5s, 5l, 5m) for binary fish detection, trained on a variety of datasets) (2023, [code](https://github.com/warplab/megafishdetector))
* [Fishial.AI fish detector](https://github.com/fishial/fish-identification?tab=readme-ov-file#models) (Mask-RCNN segmentation model and a ResNet-18 classifier for 289 classses, PyTorch) (2022, [code](https://github.com/fishial/fish-identification), [project home](https://www.fishial.ai))
* [KakaduFishAI](https://zenodo.org/records/7250921/files/202210-KakaduFishAI-CompactModel.zip?download=1) (TF object detection model for ~20 Australian species trained via Custom Vision) (2022, [code](https://github.com/ajansenn/KakaduFishAI), [data](https://zenodo.org/record/7250921#.Y_w4tMJBzl0)) 
* [NOAA Puget Sound Nearshore Fish detector](https://github.com/agentmorris/noaa-fish/releases) (2023, YOLOv5x6  trained on fish and crustaceans in the NE Pacific) ([code](https://github.com/agentmorris/noaa-fish), [data](https://lila.science/datasets/noaa-puget-sound-nearshore-fish))
* [YOLO-Fish](https://drive.google.com/drive/folders/1BmBdxwGCH3IS0kTeDxK2hT8vVvEtd_3o) (YOLOv3/v4 trained a variety of habitats) (2022, [code](https://github.com/tamim662/YOLO-Fish), [paper](https://www.sciencedirect.com/science/article/abs/pii/S1574954122002977))


### Just missing this list on technicalities...

* [Salmon Computer Vision Project](https://github.com/Salmon-Computer-Vision/salmon-computer-vision?tab=readme-ov-file) (2022, [paper](https://www.frontiersin.org/articles/10.3389/fmars.2023.1200408/full))... data and training code are available, but model weights aren't published.
* The [FathomNet model zoo](https://github.com/fathomnet/models) includes a number of detectors; I included one on the list above that's clearly for fish, but a number of other detectors for smaller marine life are included in the model zoo.
