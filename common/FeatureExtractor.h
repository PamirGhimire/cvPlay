#pragma once
#include "opencv2/features2d.hpp" 

namespace vo
{
namespace common
{
namespace visual_features
{

class FeatureExtractor
{
public:
    FeatureExtractor();
    std::vector<cv::KeyPoint> GetORBKeypointsInImage(const cv::Mat& input_image);
    std::vector<cv::KeyPoint> GetFASTKeypointsInImage(const cv::Mat& input_image);

private:
    cv::Ptr<cv::ORB> orb_feature_detector_;
    cv::Ptr<cv::FastFeatureDetector> fast_feature_detector_;
};

}//namespace visual_features ends
}//namespace common ends
}//namespace vo ends