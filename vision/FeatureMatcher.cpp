#include <FeatureMatcher.h>
#include <iostream>

namespace cvp {
namespace vision {
namespace visual_features {

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

std::vector<cv::DMatch> FilterMatchesUsingRatioTest(
    const std::vector<std::vector<cv::DMatch>> &unfiltered_knn_matches,
    const float ratio_threshold = 0.7) {

  std::vector<cv::DMatch> good_matches;

  for (const auto &unfiltered_knn_match : unfiltered_knn_matches) {
    if (!unfiltered_knn_match.empty()) {
      if (RatioTest(unfiltered_knn_match)) {
        good_matches.push_back(unfiltered_knn_match[0]);
      }
    }
  }

  return good_matches;
}

std::vector<std::vector<cv::DMatch>>
FeatureMatcher::FindKNearestNeighbourMatchesBetweenDescriptors(
    const cv::Mat &image_a_descriptors, const cv::Mat &image_b_descriptors,
    const int k) const {
  std::vector<std::vector<cv::DMatch>> knn_matches;
  orb_feature_matcher_.knnMatch(image_a_descriptors, image_b_descriptors,
                                knn_matches, k);
  return knn_matches;
}

void GetRetainedKeypointsInImagesUsingGoodMatches(
    const std::vector<cv::KeyPoint> &image_a_keypoints,
    const std::vector<cv::KeyPoint> &image_b_keypoints,
    const std::vector<cv::DMatch> &good_matches_in_image_b_for_a,
    std::vector<cv::KeyPoint> &retained_keypoints_in_image_a, // output
    std::vector<cv::KeyPoint> &retained_keypoints_in_image_b) // output
{
  if (image_a_keypoints.empty() || image_b_keypoints.empty() ||
      good_matches_in_image_b_for_a.empty())
    return;

  retained_keypoints_in_image_a.reserve(good_matches_in_image_b_for_a.size());
  retained_keypoints_in_image_b.reserve(good_matches_in_image_b_for_a.size());

  for (const auto &match : good_matches_in_image_b_for_a) {
    const auto &image_a_keypoint = image_a_keypoints[match.queryIdx];
    retained_keypoints_in_image_a.push_back(image_a_keypoint);
    const auto &image_b_keypoint = image_b_keypoints[match.trainIdx];
    retained_keypoints_in_image_b.push_back(image_b_keypoint);
  }
}

CorrespondingKeypointsInImagePair
FeatureMatcher::FindCorrespondencesBetweenTwoImages(
    const cv::Mat &input_image_a, const cv::Mat &input_image_b) const {

  const auto image_a_keypoints_and_descriptors =
      feature_extractor_.GetORBKeypointsAndDescriptorsFromImage(input_image_a);

  const auto image_b_keypoints_and_descriptors =
      feature_extractor_.GetORBKeypointsAndDescriptorsFromImage(input_image_b);

  std::vector<cv::KeyPoint> retained_keypoints_in_image_a;
  std::vector<cv::KeyPoint> retained_keypoints_in_image_b;

  if (!image_a_keypoints_and_descriptors.descriptors_.empty() &&
      !image_b_keypoints_and_descriptors.descriptors_.empty()) {

    const auto knn_matches = FindKNearestNeighbourMatchesBetweenDescriptors(
        image_a_keypoints_and_descriptors.descriptors_,
        image_b_keypoints_and_descriptors.descriptors_);

    const auto good_matches_in_image_b_for_a =
        FilterMatchesUsingRatioTest(knn_matches);

    GetRetainedKeypointsInImagesUsingGoodMatches(
        image_a_keypoints_and_descriptors.keypoints_,
        image_b_keypoints_and_descriptors.keypoints_,
        good_matches_in_image_b_for_a,
        retained_keypoints_in_image_a,  // output
        retained_keypoints_in_image_b); // output
  }

  return {retained_keypoints_in_image_a, retained_keypoints_in_image_b};
}

} // namespace visual_features
} // namespace vision
} // namespace cvp