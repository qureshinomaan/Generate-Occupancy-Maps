# Generate-Occupancy-Maps
Using DL models and Transformations for generating occupancy maps.


## Occupancy Maps
Occupancy Grid Mapping refers to a family of computer algorithms which address the problem of generating maps from noisy and uncertain data.

## Depth Image
Some methods
* Stereo Methods.
* Monocular Methods.
* Network (PSMNET)


## Three Methods Before Occupancy Grid
Semantic Segmentation
* InPlace-ABN-Mapillary
Instance Segmentation
* Mask RCNN
Depth Image
*  PSMNET


### Input
The inputs are
* Image or scene of a self driving car.
* Resolution Size for occupancy grid.
* Size of image input.

### Output
The output consists of
* Semantic Sementation in one folder.
* Depth Image in another folder.
* Occupancy Grid in another.
* The output images should have a proper name.

### Dataset
* KITTI Dataset will be used for the project.

### Resources
* [Photogometry Lectures](https://www.youtube.com/watch?v=_mOG_lpPnpY&list=PLgnQpQtFTOGRsi5vzy9PiQpNWHjq-bKN1)

### Some Other Tasks
* [Sequence to Sequence model.](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [Time Series Prediction.](https://github.com/pytorch/examples/tree/master/time_sequence_prediction)
* ResNet 34/ Resnet 50 Implementations.
