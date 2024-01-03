#ifndef __DATALOADER__
#define __DATALOADER__

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <stdexcept>

// Unix header for folder operations
#include <dirent.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

/**
 * @brief Dataloader class responsible to load parameters,
 * poses and images given their storage location and naming
 * convention.
 *
 */
class Dataloader {

private:
  cv::Matx33f K_;

  std::string path_;
  std::string img_prefix_;
  std::string img_extension_;
  const unsigned int digits_;
  unsigned int N_;

public:

  /**
   * @brief Construct a new Dataloader object
   * 
   * @param path Path to dataset
   * @param img_prefix prefix of images
   * @param digits number of digits in filename
   */
  Dataloader(std::basic_string<char> path, std::basic_string<char> img_prefix,
             std::basic_string<char> img_extension, const unsigned int digits, unsigned int N);

  /**
   * @brief returns the parsed K matrix in
   * path/K.txt
   * 
   * @return cv::Mat 3x3 K matrix
   */
  cv::Mat getK();

  /**
   * @brief Get the image of the requested index
   * 
   * @param idx Index of the requested image
   * @return cv::Mat grayscale image
   */
  cv::Mat getImg(const int idx);

  /**
   * @brief Get the pose of the requested index
   * 
   * @param idx Pose of the requested image
   * @return cv::Mat 3x4 matrix
   */
  cv::Mat getPose(const int idx);

  unsigned int size();

};

#endif