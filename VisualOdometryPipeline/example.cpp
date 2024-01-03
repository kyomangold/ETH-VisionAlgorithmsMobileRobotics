#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main() {

  cv::Mat img0 =
      cv::imread("../data/kitti05/images/000000.png", cv::IMREAD_GRAYSCALE);

  cv::Mat img1 =
      cv::imread("../data/kitti05/images/000001.png", cv::IMREAD_GRAYSCALE);

  // img *= 5;

  // img = img(cv::Rect(100,100,200,200));

  // img = img.t();

  // std::cout cv::Mat works
  //std::cout << img0 << std::endl;

  // Inefficient element access
  std::cout << (int)img0.at<uchar>(0, 0) << std::endl;

  // Very efficient access
  uchar *img_ptr = img0.ptr<uchar>(0);
  std::cout << (int)img_ptr[0] << std::endl;

  // example viz
  cv::Mat drawing(500, 500, CV_8UC3, cv::Scalar::all(255));

  cv::circle(drawing, cv::Point(250, 250), 10, cv::Scalar(0, 0, 255), 10,
             cv::FILLED);

  // example feature extractor
  // auto detector = cv::SIFT::create(300);
  // std::vector<cv::KeyPoint> keypoints0;
  // cv::Mat descriptors0;
  // detector->detectAndCompute(img0,cv::noArray(),keypoints0,descriptors0);
  // std::vector<cv::KeyPoint> keypoints1;
  // cv::Mat descriptors1;
  // detector->detectAndCompute(img1,cv::noArray(),keypoints1,descriptors1);
  // cv::drawKeypoints(img0, keypoints0, img0);

  // example matching
  // cv::Mat outputimg;
  // auto matcher = cv::BFMatcher::create(cv::NORM_L2SQR,true);
  // std::vector<cv::DMatch> matches;
  // matcher->match(descriptors0,descriptors1,matches);

  // cv::drawMatches(img0,keypoints0,img1,keypoints1,matches,outputimg);

  cv::namedWindow("Visualization");
  cv::imshow("Visualization", img0);

  cv::waitKey();

  return 0;
}