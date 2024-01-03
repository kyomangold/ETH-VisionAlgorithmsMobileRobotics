#include "viewer.hpp"
#include <ctime>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/viz/widgets.hpp>
#include <sstream>

void keyboardCallback(const cv::viz::KeyboardEvent &w, void *ptr) {
  Viewer *viewer_ = (Viewer *)ptr;
  if (w.action == cv::viz::KeyboardEvent::KEY_DOWN) {
    switch (w.code) {
    case 'f':
      viewer_->toggleFollow();
      break;
    }
  }
}

Viewer::Viewer(Map *map, const int flags) : map_(map) {

  follow_ = flags & FOLLOW;
  headless_ = flags & HEADLESS;
  if (headless_) {
    std::cout << "Viewer: Running in headless mode." << std::endl;
    window_.setOffScreenRendering();
  }

  window_.setWindowSize(cv::Size(1280, 720));
  window_.setViewerPose(
      cv::Affine3f::Identity().translate(cv::Vec3f(0, 0, -10)));
  window_.registerKeyboardCallback(keyboardCallback, this);

  cloud_ = new cv::viz::WCloud(cv::Mat(1, 1, CV_32FC3, cv::Scalar(0, 0, 0)));
  trajectory_ = new cv::viz::WTrajectory(
      std::vector<cv::Affine3d>{cv::Affine3d::Identity()});
  img_ = new cv::viz::WImageOverlay(
      cv::Mat(100, 100, CV_8UC3),
      cv::Rect(cv::Point(0, 0), window_.getWindowSize() / 3));

  if (flags & RECORD) {
    std::time_t t = time(0);
    std::tm *tm = std::localtime(&t);
    std::stringstream sfilename;
    sfilename << "./output-" << tm->tm_year + 1900 << "-" << std::setw(2)
              << std::setfill('0') << tm->tm_mon + 1 << "-" << std::setw(2)
              << std::setfill('0') << tm->tm_mday << "-" << std::setw(2)
              << std::setfill('0') << tm->tm_hour << "-" << std::setw(2)
              << std::setfill('0') << tm->tm_min << ".avi";
    // If video recording is too fast change framerate to a lower value
    recorder_.open(sfilename.str(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                   10, window_.getWindowSize());
  }
}

Viewer::~Viewer() {
  delete cloud_;
  delete trajectory_;
  delete img_;
  recorder_.release();
}

void Viewer::spin() {

  while (checkStatus()) {

    if (newdata_) {

      // #ifndef NDEBUG
      // std::cout << "Rendering" << std::endl;
      // std::cout << "Points:" << std::endl;
      // std::cout << map_->getKeyPoints() << std::endl;
      // #endif

      mutex_data_.lock();
      newdata_ = false;
      mutex_data_.unlock();

      window_.removeAllWidgets();
      delete cloud_;
      delete trajectory_;
      delete img_;
      cloud_ = new cv::viz::WCloud(map_->getKeyPoints(), cv::viz::Color::red());
      auto traj_ = map_->getTrajectory();
      trajectory_ = new cv::viz::WTrajectory(traj_, 3);
      cloud_->setRenderingProperty(cv::viz::POINT_SIZE, 4.0);
      img_ = new cv::viz::WImageOverlay(
          map_->getLastImage(),
          cv::Rect(cv::Point(0, 2 * window_.getWindowSize().height / 3),
                   window_.getWindowSize() / 3));
      window_.showWidget("Keypoints", *cloud_);
      window_.showWidget("Trajectory", *trajectory_);
      window_.showWidget("Img", *img_);

      if (follow_) {
        window_.setViewerPose(cv::Affine3f(cv::Vec3f(), cv::Vec3f(0, -3, -20))
                                  .concatenate(traj_.back()));
      }

      if (headless_) {
        window_.setOffScreenRendering();
      }
    }

    window_.spinOnce(0);

    if (recorder_.isOpened()) {
      recorder_.write(window_.getScreenshot());
    }
  }
  window_.removeAllWidgets();
  window_.close();
}

void Viewer::requestShutdown() {
  mutex_run_.lock();
  run_ = false;
  mutex_run_.unlock();
}

void Viewer::newData() {

  mutex_data_.lock();
  newdata_ = true;
  mutex_data_.unlock();
}

void Viewer::toggleFollow() { follow_ = !follow_; }

bool Viewer::checkStatus() {
  std::unique_lock<std::mutex> lock(mutex_run_);
  return run_ & !window_.wasStopped();
}
