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
