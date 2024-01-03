#ifndef __VIEWER__
#define __VIEWER__

#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/viz/widgets.hpp>

#include "map.hpp"

class Viewer {
public:
  Viewer(Map *map, const int flags = 0);
  ~Viewer();

  void spin();

  void requestShutdown();
  void newData();
  void toggleFollow();
  bool checkStatus();

  enum { FOLLOW = 1, RECORD = 2, HEADLESS = 4 };

private:
  cv::viz::Viz3d window_;
  cv::viz::WCloud *cloud_;
  cv::viz::WTrajectory *trajectory_;
  cv::viz::WImageOverlay *img_;
  cv::VideoWriter recorder_;

  Map *map_;

  std::mutex mutex_data_;
  bool newdata_ = false;
  std::mutex mutex_run_;
  bool run_ = true;
  bool follow_ = false;
  bool headless_ = false;
};

#endif