#include <FeatureMatcher.h>
#include <iostream>

namespace vo
{
namespace common
{
namespace visual_features
{


FeatureMatcher::FeatureMatcher()
{
    // feature_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    // Initialization of flann for matching ORB :  
    // https://stackoverflow.com/questions/43830849/opencv-use-flann-with-orb-descriptors-to-match-features

    feature_matcher_ = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
}


std::vector<cv::DMatch> FilterMatchesUsingRatioTest(
    const std::vector<std::vector<cv::DMatch>>& unfiltered_knn_matches, 
    const float ratio_threshold=0.7)
{
    std::vector<cv::DMatch> good_matches;

    for (size_t i = 0; i < unfiltered_knn_matches.size(); ++i)
    {
        const auto k_i = unfiltered_knn_matches[i];   
        if (!k_i.empty())
        {
            if (k_i[0].distance < ratio_threshold * k_i[1].distance)
            {
                good_matches.push_back(k_i[0]);
            }
        }
    }

    return good_matches;
}


std::vector<cv::DMatch> FeatureMatcher::MatchKeypointsInImages(const cv::Mat& input_image_a, 
                                                    const cv::Mat& input_image_b)
{
    std::vector< std::vector<cv::DMatch> > knn_matches;
    const auto input_image_a_descriptors = feature_extractor_.GetORBFeaturesFromImage(input_image_a);
    const auto input_image_b_descriptors = feature_extractor_.GetORBFeaturesFromImage(input_image_b);

    std::vector<cv::DMatch> good_matches;

    if (!input_image_a_descriptors.empty() && !input_image_b_descriptors.empty())
    {
        const int k = 2;
        feature_matcher_.knnMatch(  input_image_a_descriptors, 
                                    input_image_b_descriptors, 
                                    knn_matches, 
                                    k );

        good_matches = FilterMatchesUsingRatioTest(knn_matches);
    }
    return good_matches;
}


}//namespace visual_features ends
}//namespace common ends
}//namespace vo ends