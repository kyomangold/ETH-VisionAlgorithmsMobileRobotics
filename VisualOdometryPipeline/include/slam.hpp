#include <iostream>
#include <thread>
#include <mutex>
#include <stdexcept>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/affine.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/viz.hpp>

#include "utils.hpp"
#include "dataloader.hpp"
#include "map.hpp"
#include "viewer.hpp"

class Slam{

    public: 
    Slam(Dataloader *dataloader, Map * map, Viewer * viewer, const int flags = 0);

    void init(int idximg1, int idximg2);

    void processFrame(cv::Mat img);
    void processDataset();

    bool isFinished();
    void requestShutdown();

    enum{ HEADLESS = 1};

    private:
    Map * map_;
    Dataloader * dataloader_;
    Viewer * viewer_;
    cv::Matx33d K_;
    unsigned int rootIdx_;
    bool shutdown_ = false;
    std::mutex mshutdown_;
    bool finished_ = false;
    std::mutex mfinished_;
    bool headless_ = false;


    /**
     * @brief Previously processed image.
     * 
     */
    cv::Mat prev_img_;

    Keyframe last_keyframe_;

};