#pragma once
#include <FeatureExtractor.h>
#include <optional>

namespace cvp {
namespace vision {
namespace visual_features {

struct ImagePairFeatureCorrespondences {
  ImagePairFeatureCorrespondences(
      std::vector<cv::KeyPoint> keypoints_left,
      std::vector<cv::KeyPoint> keypoints_right,
      std::vector<cv::DMatch> matches_in_dest_right_for_left)
      : keypoints_left_image_(keypoints_left),
        keypoints_right_image_(keypoints_right),
        matches_in_dest_right_for_left_(matches_in_dest_right_for_left) {}

  std::vector<cv::KeyPoint> keypoints_left_image_;
  std::vector<cv::KeyPoint> keypoints_right_image_;
  std::vector<cv::DMatch> matches_in_dest_right_for_left_;
};

class FeatureMatcher {
public:
  FeatureMatcher();

  ImagePairFeatureCorrespondences
  FindCorrespondencesBetweenTwoImages(const cv::Mat &input_image_a,
                                      const cv::Mat &input_image_b);

private:
  std::vector<std::vector<cv::DMatch>>
  FindKNearestNeighbourMatchesBetweenDescriptors(
      const cv::Mat &image_a_descriptors, const cv::Mat &image_b_descriptors,
      const int k = 2);

private:
  FeatureExtractor feature_extractor_;
  cv::FlannBasedMatcher orb_feature_matcher_;
};

} // namespace visual_features
} // namespace vision
} // namespace cvp