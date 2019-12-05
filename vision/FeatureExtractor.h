#pragma once
#include "opencv2/features2d.hpp"

namespace cvp {
namespace vision {
namespace visual_features {

enum class VisualFeatures { FAST, ORB };

class FeatureExtractor {
public:
  FeatureExtractor();

  std::vector<cv::KeyPoint>
  GetKeypointsInImage(const cv::Mat &input_image,
                      const VisualFeatures feature_type);
  cv::Mat GetFeaturesFromImage(const cv::Mat &input_image,
                               const VisualFeatures feature_type);

  std::vector<cv::KeyPoint> GetORBKeypointsInImage(const cv::Mat &input_image);
  std::vector<cv::KeyPoint> GetFASTKeypointsInImage(const cv::Mat &input_image);
  cv::Mat GetORBFeaturesFromImage(const cv::Mat &input_image);
  cv::Mat GetFASTFeaturesFromImage(const cv::Mat &input_image);

private:
  cv::Ptr<cv::ORB> orb_feature_;
  cv::Ptr<cv::FastFeatureDetector> fast_feature_;
};

} // namespace visual_features
} // namespace vision
} // namespace cvp