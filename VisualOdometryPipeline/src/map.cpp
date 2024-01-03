#include "map.hpp"

cv::Mat Map::getKeyPoints(){
    keypoints_mutex_.lock();
    cv::Mat output(keypoints_,true);
    keypoints_mutex_.unlock();
    return std::move(output);
}

std::vector<cv::Affine3d> Map::getTrajectory(){
    std::vector<cv::Affine3d> output;
    keyframes_mutex_.lock();
    output.resize(keyframes_.size());
    for( int i = 0; i < keyframes_.size(); i++){
        output[i] = cv::Affine3d(keyframes_[i].r,keyframes_[i].t);
    }
    keyframes_mutex_.unlock();

    return std::move(output);
}

void Map::addKeyFrame(Keyframe frame,cv::InputArray img){
    last_img_mutex_.lock();
    last_img_ = img.getMat();
    last_img_mutex_.unlock();
    keyframes_mutex_.lock();
    keyframes_.push_back(frame);
    keyframes_mutex_.unlock();
}

cv:: Mat Map::getLastImage(){
    std::unique_lock<std::mutex> lock(last_img_mutex_);
    return last_img_;
}

Keyframe Map::getKeyFrame(const int i){
    std::unique_lock<std::mutex> lock(keyframes_mutex_);
    return keyframes_[i];
}

cv::Mat Map::registerKeyPoints(cv::InputArray newkeypoints){

    cv::Mat newkeypoints_ = newkeypoints.getMat();
    
    cv::Point3f* keypts_ptr = newkeypoints_.ptr<cv::Point3f>(0);

    keypoints_mutex_.lock();
    int idx_first_keyframe = keypoints_.size();
    for(int i = 0; i < newkeypoints_.rows; i++){    
        keypoints_.push_back(keypts_ptr[i]);
    }
    keypoints_mutex_.unlock();

    cv::Mat new_ids(newkeypoints_.rows,1,CV_32S);
    for(int i = 0; i < newkeypoints_.rows; i++){
        new_ids.at<int>(i) = idx_first_keyframe++;
    }

    return std::move(new_ids);
}

cv::Mat Map::requestKeyPoints(cv::InputArray requestedkeypoints){
    cv::Mat requestedkeypoints_ = requestedkeypoints.getMat();

    cv::Mat output(requestedkeypoints_.rows,3,CV_32F);
    cv::Point3f *output_ptr = output.ptr<cv::Point3f>(0);
    int * request_ptr = requestedkeypoints_.ptr<int>(0);
    keypoints_mutex_.lock();
    for(int i = 0; i < output.rows; i++){
        output_ptr[i] = keypoints_[request_ptr[i]];
    }
    keypoints_mutex_.unlock();

    return std::move(output);
}

void Map::dumpTrajectory()
{
    std::ofstream file("poses.txt", std::ios::app);

    for (auto frame : keyframes_)
    {
        file << cv::format(cv::Mat(cv::Affine3d(frame.r, frame.t).matrix).reshape(1, 1), cv::Formatter::FMT_CSV) << std::endl;
    }

    file.close();
}