#include "StereoGeometer.h"
#include "opencv2/calib3d.hpp"

namespace cvp {
namespace vision {
namespace geometry {

std::vector<cv::Point2f> GetImagePointsFromImageKeypoints(const std::vector<cv::KeyPoint> &image_keypoints)
{
  std::vector<cv::Point2f> image_points;
  image_points.reserve(image_keypoints.size());

  for(const auto& image_keypoint : image_keypoints)
  {
    image_points.emplace_back(image_keypoint.pt);
  }
  return image_points;
}

cv::Mat StereoGeometer::EstimateFundamentalMatrixFromStereoMatches(
    const std::vector<cv::KeyPoint> &image_a_keypoints,
    const std::vector<cv::KeyPoint> &image_b_keypoints) {
  
  const auto image_a_points = GetImagePointsFromImageKeypoints(image_a_keypoints);
  const auto image_b_points = GetImagePointsFromImageKeypoints(image_b_keypoints);
  try
  {
    const cv::Mat fundmat =
        (cv::Mat)cv::findFundamentalMat(image_a_points, image_b_points, cv::FM_7POINT);
    return fundmat;
  }
  catch(std::exception& e)
  {
    return cv::Mat{};
  }
}

} // namespace geometry
} // namespace vision
} // namespace cvp