import numpy as np
import open3d
import cv2



# color_raw = open3d.io.read_image("../inputs/left.png")
# depth_raw = open3d.io.read_image("../disparity.png")
# rgbd_image = open3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw);
# pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, open3d.camera.PinholeCameraIntrinsic(
#         open3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# open3d.visualization.draw_geometries([pcd])
#
# print(pcd)

cloud = open3d.io.read_point_cloud("./pointCloud.pcd")
# print(cloud.Image)
open3d.visualization.draw_geometries([cloud])
