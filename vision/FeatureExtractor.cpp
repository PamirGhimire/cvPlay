#include <FeatureExtractor.h>

namespace cvp {
namespace vision {
namespace visual_features {

FeatureExtractor::FeatureExtractor() {
  orb_feature_detector_descriptor_ = cv::ORB::create();
  fast_feature_detector_descriptor_ = cv::FastFeatureDetector::create();
}

KeypointsAndDescriptors
FeatureExtractor::GetORBKeypointsAndDescriptorsFromImage(
    const cv::Mat &input_image) const {
  std::vector<cv::KeyPoint> input_image_orb_keypoints;
  cv::Mat input_image_orb_descriptors;
  orb_feature_detector_descriptor_->detectAndCompute(
      input_image, cv::noArray(), input_image_orb_keypoints,
      input_image_orb_descriptors);

  return {input_image_orb_keypoints, input_image_orb_descriptors};
}

KeypointsAndDescriptors
FeatureExtractor::GetFASTKeypointsAndDescriptorsFromImage(
    const cv::Mat &input_image) const {
  std::vector<cv::KeyPoint> input_image_fast_keypoints;
  cv::Mat input_image_fast_descriptors;
  fast_feature_detector_descriptor_->detectAndCompute(
      input_image, cv::noArray(), input_image_fast_keypoints,
      input_image_fast_descriptors);

  return {input_image_fast_keypoints, input_image_fast_descriptors};
}

} // namespace visual_features
} // namespace vision
} // namespace cvp