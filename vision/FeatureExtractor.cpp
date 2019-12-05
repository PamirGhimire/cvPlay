#include <FeatureExtractor.h>

namespace cvp {
namespace vision {
namespace visual_features {

FeatureExtractor::FeatureExtractor() {
  orb_feature_ = cv::ORB::create();
  fast_feature_ = cv::FastFeatureDetector::create();
}

std::vector<cv::KeyPoint>
FeatureExtractor::GetORBKeypointsInImage(const cv::Mat &input_image) {
  std::vector<cv::KeyPoint> keypoints_D;

  orb_feature_->detect(input_image, keypoints_D, cv::Mat());
  return keypoints_D;
}

std::vector<cv::KeyPoint>
FeatureExtractor::GetFASTKeypointsInImage(const cv::Mat &input_image) {
  std::vector<cv::KeyPoint> keypoints_D;

  fast_feature_->detect(input_image, keypoints_D, cv::Mat());
  return keypoints_D;
}

cv::Mat FeatureExtractor::GetORBFeaturesFromImage(const cv::Mat &input_image) {
  std::vector<cv::KeyPoint> input_image_orb_keypoints;
  cv::Mat input_image_orb_descriptors;
  orb_feature_->detectAndCompute(input_image, cv::noArray(),
                                 input_image_orb_keypoints,
                                 input_image_orb_descriptors);

  return input_image_orb_descriptors;
}

cv::Mat FeatureExtractor::GetFASTFeaturesFromImage(const cv::Mat &input_image) {
  std::vector<cv::KeyPoint> input_image_fast_keypoints;
  cv::Mat input_image_fast_descriptors;
  fast_feature_->detectAndCompute(input_image, cv::noArray(),
                                  input_image_fast_keypoints,
                                  input_image_fast_descriptors);

  return input_image_fast_descriptors;
}

std::vector<cv::KeyPoint>
FeatureExtractor::GetKeypointsInImage(const cv::Mat &input_image,
                                      const VisualFeatures feature_type) {
  switch (feature_type) {
  case VisualFeatures::FAST: {
    return GetFASTKeypointsInImage(input_image);
  };
  case VisualFeatures::ORB: {
    return GetORBKeypointsInImage(input_image);
  };
  }
}

cv::Mat
FeatureExtractor::GetFeaturesFromImage(const cv::Mat &input_image,
                                       const VisualFeatures feature_type) {
  switch (feature_type) {
  case VisualFeatures::FAST: {
    return GetFASTFeaturesFromImage(input_image);
  };
  case VisualFeatures::ORB: {
    return GetORBFeaturesFromImage(input_image);
  };
  }
}

} // namespace visual_features
} // namespace vision
} // namespace cvp