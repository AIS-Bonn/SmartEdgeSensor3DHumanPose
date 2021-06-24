# SmartEdgeSensor3DHumanPose
In this repo we will publish the code for the paper:<br>
"Real-Time Multi-View 3D Human Pose Estimation using Semantic Feedback to Smart Edge Sensors"

## Citation
If you use this code for your research, please cite the following paper:

Simon Bultmann and Sven Behnke<br>
*Real-Time Multi-View 3D Human Pose Estimation using Semantic Feedback to Smart Edge Sensors*<br>
Accepted for Robotics: Science and Systems (RSS), July 2021.

## Installation
### Dependencies
The code was tested with ROS melodic and Ubuntu 18.04.

The `pose_prior` package depends on the [gtsam library](https://github.com/borglab/gtsam).
You can install it as follows (outside of catkin workspace)
```
git clone https://github.com/borglab/gtsam.git
cd gtsam
git checkout tags/4.0.3
mkdir build
cd build
cmake ..
make
sudo make install
```

### ROS packages
clone this repo inside your catkin workspace:
```
cd catkin_ws/src
git clone https://github.com/AIS-Bonn/SmartEdgeSensor3DHumanPose.git
catkin build --cmake-args -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

## Demo
get the sample data from [here](TODO).\
start `rqt` with the included perspective `pose_hall.perspective`.\
start `rviz` with the included perspective `pose_hall.rviz`.\
run the launchfile: `roslaunch pose_prior pose_triangulate_demo.launch`.\
playback the bag file: `rosbag play poses2D_16cam.bag`.

The 2D poses should be rendered in the rqt windows and the estimated triangulated 3D skeletons be displayed rviz.

## Credits
We use code from other open-source software libraries in our implementation:

The `skeleton_3d` package extends parts of the [OpenPose 3D library](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/advanced/3d_reconstruction_module.md) for multi-view triangulation.\
Cao, Zhe, et al. "OpenPose: realtime multi-person 2D pose estimation using Part Affinity Fields." IEEE transactions on pattern analysis and machine intelligence 43.1 (2019): 172-186.

The multi-person data association is based on: [Tanke, Julian, and Juergen Gall. "Iterative greedy matching for 3d human pose tracking from multiple views." German Conference on Pattern Recognition, 2019](https://github.com/jutanke/mv3dpose).

We use a public implementation of the hungarian algorithm from [here](https://github.com/mcximing/hungarian-algorithm-cpp).
