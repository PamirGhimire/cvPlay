#include "StereoGeometer.h"
#include "opencv2/calib3d.hpp"

namespace cvp {
namespace vision {
namespace geometry {

std::vector<cv::Point2f> GetImagePointsFromImageKeypoints(
    const std::vector<cv::KeyPoint> &image_keypoints) {
  std::vector<cv::Point2f> image_points;
  image_points.reserve(image_keypoints.size());

  for (const auto &image_keypoint : image_keypoints) {
    image_points.emplace_back(image_keypoint.pt);
  }
  return image_points;
}

cv::Mat StereoGeometer::EstimateFundamentalMatrixFromStereoMatches(
    const std::vector<cv::KeyPoint> &image_a_keypoints,
    const std::vector<cv::KeyPoint> &image_b_keypoints) {

  const auto image_a_points =
      GetImagePointsFromImageKeypoints(image_a_keypoints);
  const auto image_b_points =
      GetImagePointsFromImageKeypoints(image_b_keypoints);
  try {
    const cv::Mat fundmat = (cv::Mat)cv::findFundamentalMat(
        image_a_points, image_b_points, cv::FM_7POINT);
    return fundmat;
  } catch (std::exception &e) {
    return cv::Mat{};
  }
}

RigidTransformation
StereoGeometer::GetRotationAndTranslationFromEssentialMatrix(
    const cv::Mat &essential_mat) {

  cv::SVD svd_solver{};
  cv::SVD u_s_vt = svd_solver(essential_mat);
  const auto translation_upto_scale = u_s_vt.vt.row(2); // last row of vt

  cv::Mat w(3, 3, CV_64F);
  w << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  const auto R1 = u_s_vt.u * w * u_s_vt.vt;
  const auto R2 = u_s_vt.u * w.t() * u_s_vt.vt;

  if (cv::determinant(R1) < 0)
    R1 = -1 * R1;
  if (cv::determinant(R2) < 0)
    R2 = -1 * R2;
}

//@TODO: implement
cv::Mat StereoGeometer::GetCameraCalibrationMatrix() const {
  const auto focal_length_in_pixels = 500;
  return focal_length_in_pixels * cv::Mat::eye(3, 3, CV_64F);
}

} // namespace geometry
} // namespace vision
} // namespace cvp