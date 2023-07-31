# Voxel Carving for Games
## Installation
To install OpenCV:
- git clone https://github.com/opencv/opencv.git
- git clone https://github.com/opencv/opencv_contrib.git
- Create a build folder
- Open the opencv folder in CMakeGUI.
- For OPENCV_EXTRA_MODULES_PATH select {opencv_contrib_folder}/modules
- Build
- Open the OpenCV project and build the ALL_BUILD and INSTALL projects.
- On Windows, search for "Edit the System Environment Variables"
- - Click Environment Variables
  - In System Variables select Path and press edit.
  - Add the following path: absolute_path_to_opencv_build\bin\Debug
  - Restart the machine.

# How to setup the Voxel Grid
We assume a ArUco board size of 14cm * 21,5cm.  
- zSizeCM: Set the height of the voxel grid.
- voxelPerCM: Set the number of voxels by cm. (2,3,4... higher values can lead to very long execution times but improve accuracy and quality)

# How to run Voxel Carving
- #define RUN_VOXEL_CARVING 1
  
There are some parameter that should be chosen based on the video and setup:
- NUM_PROCESSED_FRAMES: [0,'MaxVideoFrames']  
  The number of camera angles taken into account in the voxel carving process.  
- DARK_THRESHOLD: [0,255]  
  We are using grayscale images to distinguish background from object pixels. A low value leads to taking only very dark pixels into account. A value too low will make the carver carve parts of the object. A high value will make the carver take brighter pixels into account which can increase accuracy. A value thats too high will make the carver miss background pixels.
- INCONSISTENCY_THRESHOLD_PERCENTAGE: [0,1]  
  When comparing color values from different frames that fall into one voxel we only carve the voxel if a certain percentage of colors are under the DARK_THRESHOLD. A low value means only a few frames can impact the carving result. A high value means more frames need to have color values over the DARK_THRESHOLD for the voxel to be carved. Due to camera estimation accuracy and grayscale threshold inaccuracy a higher value is prefered.
  The number of frames also impacts the value choice.
  A good starting percentage is 0.8.
- MULTI_SWEEP_INCONSISTENCY_THRESHOLD_PERCENTAGE: [0,1]  
  Same principle as in INCONSISTENCY_THRESHOLD_PERCENTAGE but only for the Multi-Sweep step which takes into account every camera from any direction. Until now there was no case where a value difference to INCONSISTENCY_THRESHOLD_PERCENTAGE made a difference.
- CAMERA_ABOVE_PLANE_THRESHOLD: [0,1]  
  When carving slices of the grid we choose cameras that are in front of the carving plane. (CameraPos.dot(voxel_planeNormal) > CAMERA_ABOVE_PLANE_THRESHOLD). higher values mean that only cameras that are closer to the normal of the current voxel and carving plane. Choose this value depending on the video input and your camera positioning. Default is 0.
