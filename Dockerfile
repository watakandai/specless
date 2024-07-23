FROM hardikparwana/cuda12-ubuntu22:cuda122-cudnn8-jax

# ROS2 installation
RUN apt update && apt install -y locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN LANG=en_US.UTF-8
RUN apt-get install -y software-properties-common
RUN add-apt-repository universe
RUN  apt update && apt install curl -y
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt update
ARG DEBIAN_FRONTEND=noninteractive
RUN apt install -y ros-humble-desktop
RUN apt install -y ros-dev-tools
RUN echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc

RUN apt-get install -y ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-turtlebot3*
RUN apt-get install -y ros-humble-nav2-simple-commander

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /usr/share/gazebo/setup.sh" >> ~/.bashrc
RUN echo "source /home/mobile_arm/colcon_ws/install/local_setup.bash" >> ~/.bashrc
