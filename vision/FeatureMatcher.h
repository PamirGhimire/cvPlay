#pragma once
#include <FeatureExtractor.h>
#include <exception>

namespace cvp {
namespace vision {
namespace visual_features {

struct CorrespondingKeypointsInImagePair {
  CorrespondingKeypointsInImagePair(std::vector<cv::KeyPoint> keypoints_left,
                                  std::vector<cv::KeyPoint> keypoints_right)
      : keypoints_left_image_(keypoints_left),
        keypoints_right_image_(keypoints_right) {
    if (keypoints_left.size() != keypoints_right.size())
      throw std::logic_error("Must specify a matching keypoint in image b for "
                             "every keypoint in image a for a correspondence");
  }

  bool IsValid() const {
    if (!keypoints_left_image_.empty() && !keypoints_right_image_.empty() &&
        keypoints_left_image_.size() == keypoints_right_image_.size())
      return true;

    return false;
  }

  std::vector<cv::KeyPoint> keypoints_left_image_;
  std::vector<cv::KeyPoint> keypoints_right_image_;
};

class FeatureMatcher {
public:
  FeatureMatcher();

  CorrespondingKeypointsInImagePair
  FindCorrespondencesBetweenTwoImages(const cv::Mat &input_image_a,
                                      const cv::Mat &input_image_b) const;

private:
  std::vector<std::vector<cv::DMatch>>
  FindKNearestNeighbourMatchesBetweenDescriptors(
      const cv::Mat &image_a_descriptors, const cv::Mat &image_b_descriptors,
      const int k = 2) const;

private:
  FeatureExtractor feature_extractor_;
  cv::FlannBasedMatcher orb_feature_matcher_;
};

} // namespace visual_features
} // namespace vision
} // namespace cvp