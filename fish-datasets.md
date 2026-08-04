# Datasets, models, and systems related to fish detection in marine/freshwater imagery/video

## TOC

* <a href="#publicly-available-datasets">Publicly available datasets</a>
* <a href="#publicly-available-models-for-fish-detection">Publicly-available models for fish detection</a>
* <a href="#systems-with-closed-weights-fish-detection-models">Systems with (closed-weights) fish detection models</a>


## Publicly available datasets

This section has been moved to [its own page](https://github.com/filippovarini/community-fish-detector-dataset).


## Publicly-available models for fish detection

This section lists ML models one can download and run locally on images/video of fish (or use in cloud-based systems).  This section does not include models that exist in online platforms but can't be downloaded locally.

My hope is that this section can grow into a more structured database of models with sample code... if you want to help with that, <a href="mailto:agentmorris+fishsurvey@gmail.com">email me</a>.

I am making a very loose effort to include last-updated dates for each of these.  Those dates are meant to very loosely capture repo activity, so that if you go looking for a detector for ecosystem X, you can start with more active sources.  But I'm not digging that deep; if someone trained a detector in 2016 that is totally obsolete, but they corrected a bunch of typos in their repo in 2023, they will successfully trick my algorithm for determining the last-updated date.

When possible, the first link for each line item should get you pretty close to the model weights.

* [Community Fish Detector](https://github.com/WildHackers/community-fish-detector) (family of one-class ("fish") RF-DETR  models)
* [SharkTrack](https://github.com/filippovarini/sharktrack) (YOLOv8n detector and multi-frame tracker for sharks) ([home](https://www.fvarini.com/sharktrack)) ([paper](https://www.sciencedirect.com/science/article/pii/S1574954126001196))
* [AI for the Ocean Fish and Squid Detector](https://zenodo.org/records/7430331) (YOLOv5s, trained on 5600 images in the Eastern Pacific) (2022, [code](https://github.com/heinsense2/AIO_CaseStudy))
* [MegaFishDetector](https://github.com/warplab/megafishdetector/blob/main/MODEL_ZOO.md) (YOLOv5 models (5s, 5l, 5m) for binary fish detection, trained on a variety of datasets) (2023, [code](https://github.com/warplab/megafishdetector))
* [Fishial.AI fish detector](https://github.com/fishial/fish-identification?tab=readme-ov-file#models) (Mask-RCNN segmentation model and a ResNet-18 classifier for 289 classses, PyTorch) (2022, [code](https://github.com/fishial/fish-identification), [project home](https://www.fishial.ai))
* [KakaduFishAI](https://zenodo.org/records/7250921/files/202210-KakaduFishAI-CompactModel.zip?download=1) (TF object detection model for ~20 Australian species trained via Custom Vision) (2022, [code](https://github.com/ajansenn/KakaduFishAI), [data](https://zenodo.org/record/7250921#.Y_w4tMJBzl0)) 
* [NOAA Puget Sound Nearshore Fish detector](https://github.com/agentmorris/noaa-fish/releases) (2023, YOLOv5x6  trained on fish and crustaceans in the NE Pacific) ([code](https://github.com/agentmorris/noaa-fish), [data](https://lila.science/datasets/noaa-puget-sound-nearshore-fish))
* [YOLO-Fish](https://drive.google.com/drive/folders/1BmBdxwGCH3IS0kTeDxK2hT8vVvEtd_3o) (YOLOv3/v4 trained a variety of habitats) (2022, [code](https://github.com/tamim662/YOLO-Fish), [paper](https://www.sciencedirect.com/science/article/abs/pii/S1574954122002977))


### Just missing this list on technicalities...

* [Salmon Computer Vision Project](https://github.com/Salmon-Computer-Vision/salmon-computer-vision?tab=readme-ov-file) (2022, [paper](https://www.frontiersin.org/articles/10.3389/fmars.2023.1200408/full))... data and training code are available, but model weights aren't published.
* The [FathomNet model zoo](https://github.com/fathomnet/models) includes a number of detectors; I included one on the list above that's clearly for fish, but a number of other detectors for smaller marine life are included in the model zoo.


## Systems with (closed-weights) fish detection models

* [SeaLens](https://sealens.app/)
* [Whooshh FishlRecognition](https://www.whooshh.com/product/fishlrecognition/) (fish passage monitoring system with cameras and AI)
* [AI.Fish](https://www.ai.fish/) (detection and classification for commercial fishing vessels)
* [OnDeck AI](https://ondeckai.com/) (video analysis for fisheries monitoring and marine wildlife surveys)
* [i-Ocean AI Fish Counter](https://i-ocean-global.com/en/products/fish-counter) (AI-accelerated tools for fish counting/monitoring in transport environments, including fish counting/measurement)
* [Vaki Riverwatchter](https://www.riverwatcher.is/) (AI-enabled camera/monitoring systems installed at passages)
* [MarineSitu SaltySuite](https://marinesitu.com/products/software) ("Full-stack software for underwater monitoring: edge and cloud AI, deep analytics...")
* [Ark CV](https://arkcv.com/) ("Ark CV helps fisheries teams monitor passage around the clock—adapting to the cameras, sensors, and footage already available on site")
