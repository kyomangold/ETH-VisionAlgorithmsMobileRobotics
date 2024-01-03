#ifndef __MAP__
#define __MAP__

#include <mutex>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/core/mat.hpp>


struct Keyframe{
    cv::Vec3d r; // Rotation expressed in rodriges vector
    cv::Vec3d t; // translation to keyframe
    int n_keypoints; // number of keypoints in frame
    cv::Mat keypoints; // 2d coordinates of keypoint i
    cv::Mat id_keypoint; // id of keypoint i
};

class Map{
    public:
    cv::Mat getKeyPoints();
    std::vector<cv::Affine3d> getTrajectory();
    void addKeyFrame(Keyframe frame, cv::InputArray img = cv::noArray());
    cv::Mat getLastImage();
    Keyframe getKeyFrame(const int i);
    cv::Mat registerKeyPoints(cv::InputArray keypoints);
    cv::Mat requestKeyPoints(cv::InputArray requestedkeypoints);
    void dumpTrajectory();

    private:
    std::mutex last_img_mutex_;
    cv::Mat last_img_;
    std::mutex keyframes_mutex_;
    std::vector<Keyframe> keyframes_;
    std::mutex keypoints_mutex_;
    std::vector<cv::Point3f> keypoints_;
};

#endif