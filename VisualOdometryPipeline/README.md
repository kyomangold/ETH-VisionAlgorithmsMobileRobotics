# Vision Algorithms for Mobile Robotics Project

This is the repository for the optional project of the Vision Algorithms for Mobile Robotics lecture (autumn semester 2021/2022) taught by Prof. Dr. Davide Scaramuzza.

### Videos and Trajectories
All videos and the images of the trajectories are available in this repository (top of page, linked with badges to the artifacts of the CI pipeline). The screencasts were captured on a desktop computer with an i7 3770k processor with 4 cores @ 3.3 GHz and 16 GB RAM. The code has also been tested to build and run in addtion to the CI pipeline on:

- Fedora 35; Intel i7 8565 with 4 cores @ 1.8 GHz / 16 GB (GCC 11.2 & OpenCV 4.5.2)
- macOS 12.1; Apple M1 Max with 10 cores @ 3.2 GHz / 64 GB (Clang 13.00 & OpenCV 4.5.4)
- Ubuntu 20.04; Intel i9 9880 with 16 cores @ 2.3 GHz / 32 GB (GCC 10.3 & OpenCV 4.2.0)

## Clone
This repository stores images with git-lfs. Make sure that it is installed
before pulling the repository.
```
$ git clone 
$ cd VisualOdometryPipeline
$ git lfs pull origin main # necessary for older git versions
```

## Build
```
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make
```
## Run
```
$ ./main -f # Parking
$ ./main ../data/kitti05/ -prefix="" -d=6 -f # Kitti05
$ ./main ../data/malaga/ -prefix="img" -extension=".jpg" -f # Malaga
```

## Dependencies

Mainly tested on Ubuntu 20.04, but build-tests include Ubuntu 18.04.
The code has a build and runtime dependency of:
- Opencv >= 4.0.0 (3.2 compiles but not tested to run)
- CMake >= 3.10.0 
- Numpy & Matplotlib for plotting the trajectory

## Documentation OpenCV

https://docs.opencv.org/4.5.3/

## Future improvements
- Begin to work on possible bundle adjustment using Ceres
