#include <fstream>

#include <opencv2/core.hpp>
namespace utils
{

/**
 * @brief Removes rows of src based on the provided mask.
 * Every row with zero entry in mask will be dropped.
 * 
 * @param src Input array
 * @param mask Mask determining which rows to keep(1)/remove(0).
 * @return cv::Mat Masked array
 */
cv::Mat removeRows(cv::InputArray src, cv::InputArray mask);

    
} // namespace utils