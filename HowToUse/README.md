### How to Use

#### Dependencies
* detectron2
* pytorch 1.4.0
* numpy
* opencv
* open3d

#### Note
* Some changes were made to detectron2/utils/visualizer.py
* Please copy and use the visualizer.py present in this folder.


#### To Generate Disparity Map and Instance Segmentation
* Open GenerateOccupancyMaps Notebook in the folder.


#### To Visualize in 3d
* python3 generate_pointCloud.py
* python3 3dVis/3dVis.py

#### To Generate Occupancy Maps
* python3 generate_pointCloud.py
* python3 GenerateOccupancyMap.py
* Go to 3dVis/maps folder to View results. Try opening occupancy map with a higher number like occupancy_map14.png
