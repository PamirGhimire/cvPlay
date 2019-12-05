#pragma once
#include <FeatureExtractor.h>
#include <optional>

namespace cvp {
namespace vision {
namespace visual_features {

class FeatureMatcher {
public:
  FeatureMatcher();
  std::vector<cv::DMatch> MatchKeypointsInImages(const cv::Mat &input_image_a,
                                                 const cv::Mat &input_image_b);

private:
  FeatureExtractor feature_extractor_;
  cv::FlannBasedMatcher feature_matcher_;
};

} // namespace visual_features
} // namespace vision
} // namespace cvp