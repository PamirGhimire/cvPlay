#include <FeatureMatcher.h>

namespace vo
{
namespace common
{
namespace visual_features
{


FeatureMatcher::FeatureMatcher()
{
    feature_matcher_ = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
}

std::vector<cv::DMatch> FeatureMatcher::MatchKeypointsInImages(const cv::Mat& input_image_a, 
                                                    const cv::Mat& input_image_b)
{
    std::vector< std::vector<cv::DMatch> > knn_matches;
    const auto input_image_a_descriptors = feature_extractor_.GetORBFeaturesFromImage(input_image_a);
    const auto input_image_b_descriptors = feature_extractor_.GetORBFeaturesFromImage(input_image_b);

    const int k = 2;
    feature_matcher_->knnMatch( input_image_a_descriptors, 
                                input_image_b_descriptors, 
                                knn_matches, 
                                k );

    //-- Filter matches using the Lowe's ratio test
    const float ratio_threshold = 1.0f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); ++i)
    {
        if (knn_matches[i][0].distance < ratio_threshold * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }

    return good_matches;
}


}//namespace visual_features ends
}//namespace common ends
}//namespace vo ends