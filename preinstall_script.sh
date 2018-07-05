#!/bin/bash
# May need to uncomment and update to find current packages
# apt-get update

# Required for demo script! #
pip install scikit-video

# Add your desired packages for each workspace initialization
#          Add here!          #
# So that VideoCapture may work
pip install opencv-python

# Install Bazel
# echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
# sudo apt-get update && sudo apt-get -y install bazel


# Install tensorflow from source

# Add http://www.nasm.us/pub/nasm/releasebuilds/2.12.02/nasm-2.12.02.tar.bz2


# Install Docker
# sudo apt-get install \
#     apt-transport-https \
#     ca-certificates \
#     curl \
#     software-properties-common
# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# sudo add-apt-repository \
#    "deb [arch=amd64] https://download.docker.com/linux/ubuntu artful stable"
# sudo apt-get update
# sudo apt-get -y install docker-ce