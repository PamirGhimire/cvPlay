#include <FeatureExtractor.h>

namespace vo
{
namespace common
{
namespace visual_features
{

FeatureExtractor::FeatureExtractor()
{
    orb_feature_detector_ = cv::ORB::create();
    fast_feature_detector_ = cv::FastFeatureDetector::create();
}

std::vector<cv::KeyPoint> FeatureExtractor::GetORBKeypointsInImage(
    const cv::Mat& input_image)
{
    std::vector<cv::KeyPoint> keypoints_D;

    orb_feature_detector_->detect(input_image,keypoints_D, cv::Mat());
    return keypoints_D;
}

std::vector<cv::KeyPoint> FeatureExtractor::GetFASTKeypointsInImage(
    const cv::Mat& input_image)
{
    std::vector<cv::KeyPoint> keypoints_D;

    fast_feature_detector_->detect(input_image,keypoints_D, cv::Mat());
    return keypoints_D;
}

}//namespace visual_features ends
}//namespace common ends
}//namespace vo ends