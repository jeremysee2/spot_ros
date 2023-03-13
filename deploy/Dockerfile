FROM ros:noetic-ros-base-focal

RUN DEBIAN_FRONTEND=noninteractive apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y python3-pip git qttools5-dev ros-noetic-tf2-bullet ros-noetic-roslint ros-noetic-pcl-ros
RUN pip install cython bosdyn-client bosdyn-mission bosdyn-api bosdyn-core empy numpy
RUN bash /opt/ros/noetic/setup.bash

# Download ROS packages
RUN mkdir ~/catkin_ws/src
WORKDIR ~/catkin_ws/src

# Install ROS dependencies
WORKDIR ~/catkin_ws/
RUN rosdep install --from-paths src --ignore-src -y

# Build ROS packages
RUN catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the new packages
RUN bash ~/catkin_ws/devel/setup.bash

# Setup the environmental variables for including these in the URDF
ARG SPOT_ARM=1
ARG SPOT_PACK=1
ARG SPOT_LIDAR_MOUNT=1
ARG SPOT_VELODYNE=1
ARG SPOT_USERNAME=admin

ENTRYPOINT roslaunch spot_driver driver.launch username:=admin password:=$SPOT_PASSWORD