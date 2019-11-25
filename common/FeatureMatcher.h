#pragma once
#include <FeatureExtractor.h>

namespace vo
{
namespace common
{
namespace visual_features
{

class FeatureMatcher
{
public:
    FeatureMatcher();
    std::vector<cv::DMatch> MatchKeypointsInImages(const cv::Mat& input_image_a, 
                                                     const cv::Mat& input_image_b);

private:
    FeatureExtractor feature_extractor_;
    cv::Ptr<cv::DescriptorMatcher> feature_matcher_;
};

}//namespace visual_features ends
}//namespace common ends
}//namespace vo ends