#include <dataloader.hpp>

Dataloader::Dataloader(std::basic_string<char> path,
                       std::basic_string<char> img_prefix,
                       std::basic_string<char> img_extension,
                       const unsigned int digits,
                       unsigned int N)
    : path_(path), img_prefix_(img_prefix), img_extension_(img_extension), digits_(digits), N_(N) {

  std::ifstream file;

  // Load camera matrix
  file.open(path_ + "K.txt");
  std::string line, elem;

  for (int i = 0; i < 3; i++) {
    std::getline(file, line);
    std::stringstream sline(line);
    for (int j = 0; j < 3; j++) {
      std::getline(sline, elem, ',');
      K_(i, j) = std::stof(elem);
    }
  }

  file.close();

  cv::Mat img0 = getImg(0);
  if(img0.empty()){
    throw std::runtime_error("Dataloader: Image 0 not found. Check path.");
  }

  if(N_ == 0){
    DIR *dirp;
    struct dirent *dp;
    std::string filename;

    dirp = opendir((path+"images/").c_str());

    while((dp = readdir(dirp)) != NULL){
      filename = dp->d_name;
      if(filename.length()> 4 && (filename.substr(filename.length()-4).compare(img_extension_) == 0))
        N_++;
      }
  }
}

cv::Mat Dataloader::getK() { return cv::Mat(K_); }

cv::Mat Dataloader::getImg(const int idx) {
  std::stringstream img_name;

  img_name << img_prefix_ << std::setw(digits_) << std::setfill('0') << idx
           << img_extension_;

  
  return std::move(
      cv::imread(path_ + "images/" + img_name.str(), cv::IMREAD_GRAYSCALE));
}

cv::Mat Dataloader::getPose(const int idx) {

  cv::Matx34f pose;

  std::ifstream file;
  file.open(path_ + "poses.txt");
  assert(file.is_open());
  std::string line, elem;

  for (int i = 0; i < idx; i++)
    std::getline(file, line);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      file >> pose(i, j);
    }
  }

  file.close();

  return std::move(cv::Mat(pose));
}

unsigned int Dataloader::size(){
  return N_;
}
