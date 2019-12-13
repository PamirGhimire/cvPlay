#pragma once
#include <opencv2/opencv.hpp>

namespace cvp {
namespace vision {
namespace geometry {

class StereoGeometer {
public:
  StereoGeometer() = default;

  cv::Mat EstimateFundamentalMatrixFromStereoMatches(
      const std::vector<cv::KeyPoint> &image_a_points,
      const std::vector<cv::KeyPoint> &image_b_points);
};

} // namespace geometry
} // namespace vision
} // namespace cvp