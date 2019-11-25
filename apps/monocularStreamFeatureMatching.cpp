#include "Camera.h"
#include "FeatureMatcher.h"

void DrawFeatureMatchesOnMonocularStream()
{
    using namespace vo::common;
    Camera testcam;
    visual_features::FeatureMatcher feature_matcher;    
    visual_features::FeatureExtractor feature_extractor;

    while(true)
    {
        cv::Mat test_image_a = testcam.Capture();
        cv::Mat test_image_b = testcam.Capture();
        try
        {
            const auto feature_matches_a_b = feature_matcher.MatchKeypointsInImages(
                                                                test_image_a, 
                                                                test_image_b
                                                            );

            const auto orb_keypoints_image_a = feature_extractor.GetORBKeypointsInImage(test_image_a);
            const auto orb_keypoints_image_b = feature_extractor.GetORBKeypointsInImage(test_image_b);
            
            cv::Mat image_showing_matches;
            cv::drawMatches( 
                test_image_a, 
                orb_keypoints_image_a, 
                test_image_b, 
                orb_keypoints_image_b, 
                feature_matches_a_b, 
                image_showing_matches 
                );

            const auto window_name{"matches"};
            if(!image_showing_matches.empty())
                cv::imshow(window_name, image_showing_matches);
            
            const auto escape_key = 27;
            const auto key_press = cv::waitKey(10);
            if(key_press == escape_key)
            { 
                break;
            } 
        }
        catch(...)
        {
            //do something
        }
    }
}

int main(int argc, char* argv[])
{
    DrawFeatureMatchesOnMonocularStream();
    return EXIT_SUCCESS;
}