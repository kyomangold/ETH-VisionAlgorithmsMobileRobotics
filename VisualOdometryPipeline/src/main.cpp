#include <iostream>
#include <string>
#include <thread>

#include "dataloader.hpp"
#include "slam.hpp"
#include "utils.hpp"

#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>
#if OPENCV_VERSION > 304
#include <opencv2/core/utils/logger.hpp>
#endif

int main(int argc, char *argv[]) {

#if OPENCV_VERSION > 304
  cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
#endif

  const std::string keys =
      "{help h usage ? |                | print this message   }"
      "{@path          |../data/parking/| path to dataset      }"
      "{prefix         |img_            | prefix for images    }"
      "{extension      |.png             | fileextension for images}"
      "{digits d       |5               | prefix for images    }"
      "{count N        |                | count of images      }"
      "{f              |                | set camera to follow }"
      "{r              |                | record visualization }"
      "{headless       |            | runs application headless}";

  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("VO v1.0.0");

  if (parser.has("help")) {
    parser.printMessage();
    return 0;
  }

  if (!parser.check()) {
    parser.printErrors();
    return 0;
  }

  std::string path = parser.get<cv::String>(0);
  std::string prefix = parser.get<cv::String>("prefix");
  std::string extension = parser.get<cv::String>("extension");
  unsigned int digits = parser.get<int>("digits");
  unsigned int N;
  int viewerflags = 0;
  int slamflags = 0;
  if(parser.has("N")){
    N = parser.get<int>("N");
  }
  else{
    N = 0;
  }
  if(parser.has("f")){
    viewerflags += Viewer::FOLLOW;
  }
  if(parser.has("r")){
    viewerflags += Viewer::RECORD;
  }
    if(parser.has("headless")){
    viewerflags += Viewer::HEADLESS;
    slamflags += Slam::HEADLESS;
  }

  Dataloader dataloader(path, prefix, extension, digits, N);

  auto map = new Map();
  Viewer viewer(map,viewerflags);
  Slam slam(&dataloader,map,&viewer,slamflags);

  slam.init(0,2);

  std::thread thslam(&Slam::processDataset,&slam);

  viewer.spin();

  if(!slam.isFinished()){
    slam.requestShutdown();
  }
  thslam.join();

  map->dumpTrajectory();

  delete map;

  return 0;
}