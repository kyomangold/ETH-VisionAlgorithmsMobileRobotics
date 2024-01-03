#include <utils.hpp>

namespace utils {

cv::Mat removeRows(cv::InputArray src, cv::InputArray mask) {

  cv::Mat src_ = src.getMat();
  cv::Mat mask_ = mask.getMat();

  int rows = cv::countNonZero(mask_);

  cv::Mat dst(rows, src_.cols, src_.type());

  int i = 0;
  for (int j = 0; i < rows; j++) {
    if (mask_.at<uchar>(j))
      src_.row(j).copyTo(dst.row(i++));
  }

  return dst;
}


}