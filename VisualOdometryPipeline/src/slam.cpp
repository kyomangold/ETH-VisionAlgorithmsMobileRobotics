#include "slam.hpp"

#define DEBUG_INIT

Slam::Slam(Dataloader *dataloader, Map *map, Viewer *viewer, const int flags)
    : dataloader_(dataloader), map_(map), viewer_(viewer) {
  K_ = dataloader_->getK();
  headless_ = HEADLESS & flags;
}

void Slam::init(int idximg1, int idximg2) {
  cv::Mat img1 = dataloader_->getImg(idximg1);
  cv::Mat img2 = dataloader_->getImg(idximg2);
  rootIdx_ = idximg1;

  std::cout << "=== Initializing pipeline ===" << std::endl;
  cv::Mat corners1, corners2;
  cv::Mat keyPoints;

  cv::Mat status, error, mask, img_mask, E, R, t;

  img_mask = cv::Mat(img1.size(), CV_8U, cv::Scalar::all(0));
  img_mask(cv::Rect(img1.cols / 6, img1.rows / 6, img1.cols - img1.cols / 3,
                    img1.rows - img1.rows / 3)) = 1;

  // Determine good features to track using Shi-Tomasi detector
  cv::goodFeaturesToTrack(img1, corners1, 0, 0.001, 9, img_mask);

  std::cout << "Found " << corners1.rows << " potential features." << std::endl;

  cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 50, 0.001);

  // Track features across initialization images
  cv::calcOpticalFlowPyrLK(img1, img2, corners1, corners2, status, error,
                           cv::Size(21, 21), 5, criteria);

  // Remove untracked features
  corners1 = utils::removeRows(corners1, status);
  corners2 = utils::removeRows(corners2, status);

  std::cout << "Found " << corners1.rows << " feature correspondances."
            << std::endl;

  // Calculate 3d transformation between frames.
  E = cv::findEssentialMat(corners2, corners1, K_, cv::RANSAC, 0.9999, 1, mask);
  std::cout << "Inlier count: "
            << (float)cv::countNonZero(mask) / (float)mask.rows * 100.0 << "%"
            << std::endl;

  // Given these inputs the transformation [R|t] corresponds to
  // the transformation from coordinate system 1 to coordinate system 2
  cv::recoverPose(E, corners2, corners1, K_, R, t, mask);

  // Remove outliers
  corners1 = utils::removeRows(corners1, mask);
  corners2 = utils::removeRows(corners2, mask);

  cv::Affine3d transform(R, t);

  // Triangulate first points
  // Since we want the coordinate system of img1 to be world we need
  // the inverse transform to correctly triangulate the points.

  cv::triangulatePoints(
      cv::Mat(K_) * cv::Mat(transform.inv().matrix).rowRange(0, 3),
      cv::Mat(K_) * cv::Mat::eye(3, 4, CV_64F), corners2, corners1, keyPoints);

  keyPoints = keyPoints.t();
  cv::convertPointsFromHomogeneous(keyPoints, keyPoints);

  // Outputs basis transformation of camera pose from img1 to img2
  std::cout << "First estimated transformation:" << std::endl;
  std::cout << transform.matrix << std::endl;

  // Initialize map
  prev_img_ = std::move(img1);
  last_keyframe_.r = cv::Affine3d::Identity().rvec();
  last_keyframe_.t = cv::Affine3d::Identity().translation();
  last_keyframe_.n_keypoints = keyPoints.rows;
  last_keyframe_.keypoints = std::move(corners1);
  last_keyframe_.id_keypoint = map_->registerKeyPoints(keyPoints);

  cv::Mat output_img;
  cv::cvtColor(prev_img_, output_img, cv::COLOR_GRAY2BGR);
  for (int i = 0; i < last_keyframe_.n_keypoints; i++) {
    cv::drawMarker(output_img, last_keyframe_.keypoints.at<cv::Point2f>(i),
                   cv::Scalar(0, 0, 255));
    cv::line(output_img, last_keyframe_.keypoints.at<cv::Point2f>(i),
             corners2.at<cv::Point2f>(i), cv::Scalar(255, 0, 0));
  }

  map_->addKeyFrame(last_keyframe_, output_img);
  std::cout << "=== Successfully initialized with "
            << last_keyframe_.n_keypoints << " keypoints. ===" << std::endl;

  viewer_->newData();

#ifndef DEBUG_INIT

  cv::viz::Viz3d myWindow("Debug Initialization");

  cv::viz::WCameraPosition camera1(K_, prev_img_, 1);
  cv::viz::WCameraPosition camera1xyz(1);
  cv::viz::WCameraPosition camera2(K_, img2, 1, cv::viz::Color::red());
  cv::viz::WImageOverlay Wimg(
      output_img, cv::Rect(cv::Point(0, 0), myWindow.getWindowSize() / 2));
  camera2.applyTransform(transform);

  cv::viz::WCloud cloud(keyPoints.reshape(3), cv::viz::Color::red());
  cloud.setRenderingProperty(cv::viz::POINT_SIZE, 4.0);

  myWindow.showWidget("Points", cloud);
  myWindow.showWidget("Camera1", camera1);
  myWindow.showWidget("Cameraxyz", camera1xyz);
  myWindow.showWidget("Camera2", camera2);
  myWindow.showWidget("Image", Wimg);
  myWindow.setViewerPose(
      cv::Affine3f::Identity().translate(cv::Vec3f(0, 0, -10)));
  while (!myWindow.wasStopped()) {
    myWindow.spinOnce();
  }

#endif
}

void Slam::processFrame(cv::Mat img) {

  cv::Mat status, error, inliers;

  cv::Mat curr_2d_keypoints_;

  cv::TermCriteria criteria = cv::TermCriteria(
      cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 2000, 0.01);

  // Find new candidates:

  // create mask around existing keypoints
  cv::Mat mask(prev_img_.size(), CV_8U, cv::Scalar::all(1));
  for (int i = 0; i < last_keyframe_.n_keypoints; i++) {
    if (last_keyframe_.keypoints.at<cv::Point2f>(i).x < 5 ||
        last_keyframe_.keypoints.at<cv::Point2f>(i).y < 5 ||
        last_keyframe_.keypoints.at<cv::Point2f>(i).x > img.cols - 5 ||
        last_keyframe_.keypoints.at<cv::Point2f>(i).y > img.rows - 5) {
      continue;
    }
    mask(cv::Rect(last_keyframe_.keypoints.at<cv::Point2f>(i) -
                      cv::Point2f(4, 4),
                  cv::Size(9, 9))) = 0;
  }

  // find new keypoints
  cv::Mat new_keypoint_candidates;
  cv::goodFeaturesToTrack(img, new_keypoint_candidates,
                          std::max(1000 - last_keyframe_.n_keypoints, 1), 0.01,
                          9, mask);

  cv::Mat new_keypoint_candidates_tracked, status_candidates, error_candidates;

  // track keypoints across one frame
  cv::calcOpticalFlowPyrLK(prev_img_, img, new_keypoint_candidates,
                           new_keypoint_candidates_tracked, status_candidates,
                           error_candidates, cv::Size(21, 21), 5, criteria);

  new_keypoint_candidates =
      utils::removeRows(new_keypoint_candidates, status_candidates);
  new_keypoint_candidates_tracked =
      utils::removeRows(new_keypoint_candidates_tracked, status_candidates);

  // Track triangulated keypoints to new image
  cv::Mat last_keypoints = last_keyframe_.keypoints;
  cv::calcOpticalFlowPyrLK(prev_img_, img, last_keypoints, curr_2d_keypoints_,
                           status, error, cv::Size(21, 21), 5, criteria);

  curr_2d_keypoints_ = utils::removeRows(curr_2d_keypoints_, status);
  last_keypoints = utils::removeRows(last_keypoints, status);

  cv::Mat matched_ids = utils::removeRows(last_keyframe_.id_keypoint, status);
  cv::Mat matched_keypoints = map_->requestKeyPoints(matched_ids);

  // Estimate transformation world -> camera.
  cv::Mat r, t;
  cv::solvePnPRansac(matched_keypoints, curr_2d_keypoints_, K_, cv::noArray(),
                     r, t, false, 2000, 2.5, 0.9999, inliers);
  std::cout << "Tracked: " << inliers.rows << " points to new frame."
            << std::endl;

  if (inliers.rows <= 30) {
    throw std::runtime_error("I am lost!");
  }

  // Remove outliers from current keypoints
  cv::Mat matched_ids_tmp(inliers.rows, 1, matched_ids.type());
  cv::Mat matched_2d_keypoints(inliers.rows, curr_2d_keypoints_.cols,
                               curr_2d_keypoints_.type());
  cv::Mat matched_last_keypoints(inliers.rows, last_keypoints.cols,
                                 last_keypoints.type());
  for (int i = 0; i < inliers.rows; i++) {
    matched_ids.row(inliers.at<int>(i)).copyTo(matched_ids_tmp.row(i));
    curr_2d_keypoints_.row(inliers.at<int>(i))
        .copyTo(matched_2d_keypoints.row(i));
    last_keypoints.row(inliers.at<int>(i))
        .copyTo(matched_last_keypoints.row(i));
  }

  cv::Mat output_img;
  cv::cvtColor(prev_img_, output_img, cv::COLOR_GRAY2BGR);

  for (int i = 0; i < matched_last_keypoints.rows; i++) {
    cv::drawMarker(output_img, matched_2d_keypoints.at<cv::Point2f>(i),
                   cv::Scalar(0, 0, 255));
    cv::line(output_img, matched_2d_keypoints.at<cv::Point2f>(i),
             matched_last_keypoints.at<cv::Point2f>(i), cv::Scalar(255, 0, 0));
  }

  // cv::namedWindow("Test");
  // cv::imshow("Test",output_img);

  // Curr world -> camera
  cv::Affine3d curr(r, t);
  // Inverse of this is current position of camera.
  std::cout << "New position:" << std::endl;
  std::cout << curr.inv().matrix << std::endl;

  // Triangulate new keypoints
  cv::Mat new_keypoints;
  if (!new_keypoint_candidates_tracked.empty()) {
    cv::triangulatePoints(
        cv::Mat(K_) * cv::Mat(curr.matrix).rowRange(0, 3),
        cv::Mat(K_) *
            cv::Mat(
                cv::Affine3d(last_keyframe_.r, last_keyframe_.t).inv().matrix)
                .rowRange(0, 3),
        new_keypoint_candidates_tracked, new_keypoint_candidates,
        new_keypoints);

    new_keypoints = new_keypoints.t();
    cv::convertPointsFromHomogeneous(new_keypoints, new_keypoints);
  }
  cv::Mat new_keypoints_status(new_keypoints.rows, 1, CV_8U, cv::Scalar(1));

  // Remove bad triangulated candidates
  for (int i = 0; i < new_keypoints.rows; i++) {
    cv::Point3f test = cv::Affine3d(last_keyframe_.r, last_keyframe_.t).inv() *
                       new_keypoints.at<cv::Point3f>(i);
    if (test.z < 6 || test.z > 200) {
      new_keypoints_status.at<uchar>(i) = 0;
    }
  }

  new_keypoint_candidates_tracked =
      utils::removeRows(new_keypoint_candidates_tracked, new_keypoints_status);
  new_keypoints = utils::removeRows(new_keypoints, new_keypoints_status);

  // Update Map and Viewer
  cv::Mat new_ids = map_->registerKeyPoints(new_keypoints);

  Keyframe frame;
  frame.r = curr.inv().rvec();
  frame.t = curr.inv().translation();
  cv::vconcat(matched_ids_tmp, new_ids, frame.id_keypoint);
  cv::vconcat(matched_2d_keypoints, new_keypoint_candidates_tracked,
              frame.keypoints);
  frame.n_keypoints = matched_2d_keypoints.rows + new_keypoints.rows;
  map_->addKeyFrame(frame, output_img);
  last_keyframe_ = frame;
  prev_img_ = img;
  viewer_->newData();
}

void Slam::processDataset() {
  auto timer = std::chrono::high_resolution_clock();
  auto tstart = timer.now();
  auto tend = timer.now();
  std::chrono::duration<double> time;
  for (int i = rootIdx_ + 1; i < dataloader_->size(); i++) {
    mshutdown_.lock();
    if (shutdown_) {
      std::cout << "Slam: Shutdown requested. Terminating." << std::endl;
      return;
    }
    mshutdown_.unlock();

    std::cout << "Processing frame: " << i << std::endl;
    cv::Mat img = dataloader_->getImg(i);
    if (img.empty()) {
      std::cout << "Detected invalid image. Skipping." << std::endl;
      break;
    }
    tstart = timer.now();
    processFrame(img);
    tend = timer.now();
    time = tend - tstart;
    std::cout << "Time: " << time.count() << std::endl;
    std::cout << "FPS: " << 1.0 / time.count() << std::endl;
  }
  mfinished_.lock();
  finished_ = true;
  mfinished_.unlock();
  std::cout << "Dataset processed." << std::endl;
  if (headless_) {
    viewer_->requestShutdown();
  }
}

bool Slam::isFinished() {
  std::unique_lock<std::mutex> lock(mfinished_);
  return finished_;
}

void Slam::requestShutdown() {
  std::unique_lock<std::mutex> lock(mshutdown_);
  shutdown_ = true;
}