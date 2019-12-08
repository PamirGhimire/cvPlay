#pragma once
#include "opencv2/features2d.hpp"

namespace cvp {
namespace vision {
namespace visual_features {

enum class VisualFeatures { FAST, ORB };
struct KeypointsAndDescriptors {
  KeypointsAndDescriptors(const std::vector<cv::KeyPoint> &keypoints,
                          const cv::Mat &descriptors)
      : keypoints_(keypoints), descriptors_(descriptors) {}
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
};

class FeatureExtractor {
public:
  FeatureExtractor();

  KeypointsAndDescriptors
  GetORBKeypointsAndDescriptorsFromImage(const cv::Mat &input_image);
  KeypointsAndDescriptors
  GetFASTKeypointsAndDescriptorsFromImage(const cv::Mat &input_image);

private:
  cv::Ptr<cv::ORB> orb_feature_detector_descriptor_;
  cv::Ptr<cv::FastFeatureDetector> fast_feature_detector_descriptor_;
};

} // namespace visual_features
} // namespace vision
} // namespace cvp