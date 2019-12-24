#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

namespace cvp {
namespace vision {
namespace geometry {

struct RigidTransformation {
  RigidTransformation() = default;
  cv::Mat rotation_(3, 3, CV_64F);
  cv::Mat translation(3, 1, CV_64F);
}

class StereoGeometer {
public:
  StereoGeometer() = default;

  cv::Mat EstimateFundamentalMatrixFromStereoMatches(
      const std::vector<cv::KeyPoint> &image_a_points,
      const std::vector<cv::KeyPoint> &image_b_points);

  cv::SVD
  GetRotationAndTranslationFromEssentialMatrix(const cv::Mat &fundamental_mat);

  cv::Mat GetCameraCalibrationMatrix() const;

} // namespace geometry
} // namespace geometry
} // namespace vision