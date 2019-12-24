#include "FeatureMatcher.h"
#include "StereoGeometer.h"
#include "TimelapseCamera.h"

std::vector<cv::DMatch> GetVectorSpecifyingKMatchesToKOfSize(size_t n_matches) {
  std::vector<cv::DMatch> specify_k_in_a_matches_to_k_in_b;
  specify_k_in_a_matches_to_k_in_b.reserve(n_matches);
  for (size_t i = 0; i < n_matches; ++i) {
    constexpr const float distance_between_descriptors{0};
    specify_k_in_a_matches_to_k_in_b.emplace_back(i, i,
                                                  distance_between_descriptors);
  }
  return specify_k_in_a_matches_to_k_in_b;
}

void ShowMatchesBetweenTimeSeparatedFrames(
    const cvp::vision::TimeSeparatedFrames &time_separated_frames) {
  using namespace cvp::vision;
  cv::Mat image_left = time_separated_frames.current_frame_;
  cv::Mat image_right = time_separated_frames.delayed_frame_;

  visual_features::FeatureMatcher feature_matcher;
  const auto feature_correspondences_left_right =
      feature_matcher.FindCorrespondencesBetweenTwoImages(image_left,
                                                          image_right);

  cv::Mat image_showing_matches;

  if (feature_correspondences_left_right.IsValid()) {
    const auto keypoints_in_left_image =
        feature_correspondences_left_right.keypoints_left_image_;
    const auto keypoints_in_right_image =
        feature_correspondences_left_right.keypoints_right_image_;
    const auto feature_matches_left_right =
        GetVectorSpecifyingKMatchesToKOfSize(keypoints_in_left_image.size());

    cv::drawMatches(image_left, keypoints_in_left_image, image_right,
                    keypoints_in_right_image, feature_matches_left_right,
                    image_showing_matches);
  }

  const auto window_name{"feature matching between time-separated frames"};
  if (!image_showing_matches.empty())
    cv::imshow(window_name, image_showing_matches);
}

bool UserPressedEscapeKey() {
  const auto escape_key = 27;
  const auto key_press = cv::waitKey(10);
  if (key_press == escape_key) {
    return true;
  }
  return false;
}

cvp::vision::visual_features::CorrespondingKeypointsInImagePair
FindMatchesBetweenTimeSeparatedFrames() {
  using namespace cvp::vision;
}

cv::Mat EstimateFundamentalMatrixFromMatches(
    const cvp::vision::visual_features::CorrespondingKeypointsInImagePair
        &matches_between_images) {
  using namespace cvp::vision;
  geometry::StereoGeometer stereo_geometer;
  const auto epipolar_geometry_from_matches =
      stereo_geometer.EstimateFundamentalMatrixFromStereoMatches(
          matches_between_images.keypoints_left_image_,
          matches_between_images.keypoints_right_image_);

  return epipolar_geometry_from_matches;
}

// std::vector<cv::Mat>
// GetRotationAndTranslationFromFundamentalMat(const cv::Mat &fundamental_mat) {
//   using namespace cvp::vision;
//   geometry::StereoGeometer stereo_geometer;

//   const auto rotation_and_translation =
//       stereo_geometer.GetRotationAndTranslationFromFundamentalMatrix(
//           fundamental_mat);

//   return rotation_and_translation;
// }

void EstimateFundamentalMatrixBetweenConsecutiveFrames(
    const cvp::vision::TimeSeparatedFrames &time_separated_frames) {
  using namespace cvp::vision;

  const auto consecutive_frames = time_separated_frames;

  cv::Mat image_left = time_separated_frames.current_frame_;
  cv::Mat image_right = time_separated_frames.delayed_frame_;

  visual_features::FeatureMatcher feature_matcher;
  const auto feature_matches_left_right =
      feature_matcher.FindCorrespondencesBetweenTwoImages(image_left,
                                                          image_right);

  auto correspondences_between_images = feature_matches_left_right;
  auto fundamental_matrix =
      EstimateFundamentalMatrixFromMatches(correspondences_between_images);

  std::cout << fundamental_matrix << std::endl << std::endl;
}

void DrawFeatureMatchesOnMonocularStream() {
  using namespace cvp::vision;
  const auto number_of_frames_to_skip_between_matched_frames = 5;
  TimelapseCamera timelapse_camera(
      number_of_frames_to_skip_between_matched_frames);

  while (true) {
    timelapse_camera.Update();
    const auto time_separated_frames =
        timelapse_camera.GetTimeSeparatedFrames();

    if (time_separated_frames.IsValid()) {
      ShowMatchesBetweenTimeSeparatedFrames(time_separated_frames);
      EstimateFundamentalMatrixBetweenConsecutiveFrames(time_separated_frames);
    }
    if (UserPressedEscapeKey()) {
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  DrawFeatureMatchesOnMonocularStream();
  return EXIT_SUCCESS;
}