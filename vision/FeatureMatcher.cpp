#include <FeatureMatcher.h>
#include <iostream>

namespace cvp {
namespace vision {
namespace visual_features {

struct RetainedMatchesAfterRatioTest {
  RetainedMatchesAfterRatioTest() = default;
  RetainedMatchesAfterRatioTest(
      const std::vector<cv::DMatch> &knn_matches_passing_ratio_test,
      std::vector<unsigned int> &index_of_retained_knn_matches)
      : knn_matches_passing_ratio_test_(knn_matches_passing_ratio_test),
        indices_of_retained_knn_matches_(index_of_retained_knn_matches) {}

  std::vector<cv::DMatch> knn_matches_passing_ratio_test_;
  std::vector<unsigned int> indices_of_retained_knn_matches_;
};

FeatureMatcher::FeatureMatcher() {
  // Initialization of flann for matching ORB :
  // https://stackoverflow.com/questions/43830849/opencv-use-flann-with-orb-descriptors-to-match-features
  orb_feature_matcher_ =
      cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
}

bool RatioTest(const std::vector<cv::DMatch> &knn_match_with_k_ge_2,
               const float ratio_threshold = 0.7f) {
  return knn_match_with_k_ge_2[0].distance <
         ratio_threshold * knn_match_with_k_ge_2[1].distance;
}

RetainedMatchesAfterRatioTest FilterMatchesUsingRatioTest(
    const std::vector<std::vector<cv::DMatch>> &unfiltered_knn_matches,
    const float ratio_threshold = 0.7) {

  std::vector<cv::DMatch> good_matches;
  std::vector<unsigned int> indices_of_retained_knn_matches;

  for (unsigned int i = 0; i < unfiltered_knn_matches.size(); ++i) {
    const auto &unfiltered_knn_match = unfiltered_knn_matches[i];
    if (!unfiltered_knn_match.empty()) {
      if (RatioTest(unfiltered_knn_match)) {
        good_matches.push_back(unfiltered_knn_match[0]);
        indices_of_retained_knn_matches.push_back(i);
      }
    }
  }

  return {good_matches, indices_of_retained_knn_matches};
}

std::vector<std::vector<cv::DMatch>>
FeatureMatcher::FindKNearestNeighbourMatchesBetweenDescriptors(
    const cv::Mat &image_a_descriptors, const cv::Mat &image_b_descriptors,
    const int k) {
  std::vector<std::vector<cv::DMatch>> knn_matches;
  orb_feature_matcher_.knnMatch(image_a_descriptors, image_b_descriptors,
                                knn_matches, k);
  return knn_matches;
}

ImagePairFeatureCorrespondences
FeatureMatcher::FindCorrespondencesBetweenTwoImages(
    const cv::Mat &input_image_a, const cv::Mat &input_image_b) {

  const auto image_a_keypoints_and_descriptors =
      feature_extractor_.GetORBKeypointsAndDescriptorsFromImage(input_image_a);

  const auto image_b_keypoints_and_descriptors =
      feature_extractor_.GetORBKeypointsAndDescriptorsFromImage(input_image_b);

  std::vector<cv::DMatch> good_matches_in_image_b_for_a;
  std::vector<cv::KeyPoint> retained_keypoints_in_image_a;
  std::vector<cv::KeyPoint> retained_keypoints_in_image_b;

  RetainedMatchesAfterRatioTest retained_knn_matches;

  if (!image_a_keypoints_and_descriptors.descriptors_.empty() &&
      !image_b_keypoints_and_descriptors.descriptors_.empty()) {

    const auto knn_matches = FindKNearestNeighbourMatchesBetweenDescriptors(
        image_a_keypoints_and_descriptors.descriptors_,
        image_b_keypoints_and_descriptors.descriptors_);

    retained_knn_matches = FilterMatchesUsingRatioTest(knn_matches);
    retained_keypoints_in_image_a =
        image_a_keypoints_and_descriptors.keypoints_;
    retained_keypoints_in_image_b =
        image_b_keypoints_and_descriptors.keypoints_;
  }

  return {retained_keypoints_in_image_a, retained_keypoints_in_image_b,
          retained_knn_matches.knn_matches_passing_ratio_test_};
}

} // namespace visual_features
} // namespace vision
} // namespace cvp