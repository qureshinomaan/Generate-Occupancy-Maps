# Generate-Occupancy-Maps
Using pre-trained Deep Learning models and Transformations for generating occupancy maps.

## Note
If you find any difficulty in using some notebook or script, please feel free to create an issue. This is a work in progress and I will keep making changes to the repository for a while.

## Occupancy Maps
Occupancy Grid Mapping refers to a family of computer algorithms which address the problem of generating maps from noisy and uncertain data.

## Algorithm
The system takes a stereo pair and generates a depth map(using PSMNet) and instance segmented scene(using maskrcnn). We then use these to get a 3D Model of the scene. This 3d model is projected to the ground to get the occupancy grid.
!["Algorithmic Pipeline"](./HowToUse/pipeline.png)


## Depth Image
In CV, a depth image contains information about depth of surfaces presents in the image.
Some methods to get the depth image.
* Stereo Methods.
* Monocular Methodss.
* Network (PSMNET)
  * PSMNet actually gives disparity map, which can be converted to a depth map.

## Instance Segmentation
We identify each instance of each object featured in the image instead of categorizing each pixel like in semantic segmentation.
* We can use pretrained models of detectron2 model-zoo.
 

### Input
The inputs are
* Image or scene of a self driving car.
* Resolution Size for occupancy grid(to be done).
* Size of image input.

### Output
The output consists of
* Semantic Segmentation in one folder.
* Depth Image in another folder.
* Occupancy Grid in another.
* The output images should have a proper name.

### Dataset
* KITTI Dataset is used for testing in the project.

### Tasks
- [x] Debug PSMNet to get the depth image.
- [x] Using detectron2 to get the instance segmented image(without text).
- [x] Using depth image to obtain a 3d visualisation.
- [x] Getting occupancy maps from 3d visualisation.
- [ ] Format in the input/output format mentioned above.
- [ ] Making the occupancy map better.
- [ ] Writing scripts so that the system is easily usable.
- [ ] Adding the Monocular depth maps.
- [ ] Stitching the point clouds to get a map of the environment(a little ambitious for now).
  -  If you are a beginner and want to team up on this, please contact. 


### Other Tasks
- [x] [Time Series Prediction.](https://github.com/pytorch/examples/tree/master/time_sequence_prediction)
- [x] ResNet 34/ Resnet 50 Implementations.
- [x] GANs Implementation.
- [ ] [Sequence to Sequence model.](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

### Resources
* [Photogometry Lectures](https://www.youtube.com/watch?v=_mOG_lpPnpY&list=PLgnQpQtFTOGRsi5vzy9PiQpNWHjq-bKN1)

### Mentor
[Shashank Srikanth](https://github.com/talsperre)
