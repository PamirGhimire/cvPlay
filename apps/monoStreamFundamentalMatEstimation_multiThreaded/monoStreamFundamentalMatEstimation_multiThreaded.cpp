#include "FeatureMatcher.h"
#include "StereoGeometer.h"
#include "TimelapseCamera.h"

#include <thread>

struct AppData {
  cvp::vision::TimeSeparatedFrames time_separated_frames_;
  bool app_is_alive_{true};
};
static AppData app_data;

void UpdateDataFromTimelapseCamera() {
  using namespace cvp::vision;
  const auto number_of_frames_to_skip_between_matched_frames = 5;
  TimelapseCamera timelapse_camera(
      number_of_frames_to_skip_between_matched_frames);

  while (app_data.app_is_alive_) {
    timelapse_camera.Update();
    app_data.time_separated_frames_ = timelapse_camera.GetTimeSeparatedFrames();
  }
}

std::vector<cv::DMatch> GetVectorSpecifyingKMatchesToKOfSize(size_t n_matches) {
  std::vector<cv::DMatch> specify_k_in_a_matches_to_k_in_b;
  specify_k_in_a_matches_to_k_in_b.reserve(n_matches);
  for (size_t i = 0; i < n_matches; ++i) {
    specify_k_in_a_matches_to_k_in_b.emplace_back(i, i, 0.5);
  }
  return specify_k_in_a_matches_to_k_in_b;
}

void ShowMatchesBetweenTimeSeparatedFrames() {
  using namespace cvp::vision;
  cv::Mat image_left = app_data.time_separated_frames_.current_frame_;
  cv::Mat image_right = app_data.time_separated_frames_.delayed_frame_;

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

void DrawFeatureMatchesOnMonocularStream() {
  using namespace cvp::vision;
  const auto number_of_frames_to_skip_between_matched_frames = 5;
  TimelapseCamera timelapse_camera(
      number_of_frames_to_skip_between_matched_frames);

  while (true) {
    timelapse_camera.Update();
    app_data.time_separated_frames_ = timelapse_camera.GetTimeSeparatedFrames();

    if (app_data.time_separated_frames_.IsValid())
      ShowMatchesBetweenTimeSeparatedFrames();

    if (UserPressedEscapeKey()) {
      app_data.app_is_alive_ = false;
      break;
    }
  }
}

cv::Mat EstimateFundamentalMatrixFromMatches(
    const cvp::vision::visual_features::CorrespondingKeypointsInImagePair &matches_between_images) {
  using namespace cvp::vision;
  geometry::StereoGeometer stereo_geometer;
  const auto epipolar_geometry_from_matches =
      stereo_geometer.EstimateFundamentalMatrixFromStereoMatches(
        matches_between_images.keypoints_left_image_,
        matches_between_images.keypoints_right_image_
      );

  return epipolar_geometry_from_matches;
}

cvp::vision::visual_features::CorrespondingKeypointsInImagePair FindMatchesBetweenTimeSeparatedFrames() {
  using namespace cvp::vision;
  cv::Mat image_left = app_data.time_separated_frames_.current_frame_;
  cv::Mat image_right = app_data.time_separated_frames_.delayed_frame_;

  visual_features::FeatureMatcher feature_matcher;
  const auto feature_matches_left_right =
      feature_matcher.FindCorrespondencesBetweenTwoImages(image_left, image_right);

  return feature_matches_left_right;
}

void EstimateFundamentalMatrixBetweenConsecutiveFrames() {
  const auto consecutive_frames = app_data.time_separated_frames_;
  if (consecutive_frames.IsValid()) {
    auto correspondences_between_images = FindMatchesBetweenTimeSeparatedFrames();
    auto fundamental_matrix =
        EstimateFundamentalMatrixFromMatches(correspondences_between_images);
  
    std::cout << fundamental_matrix << std::endl << std::endl;
  }
}

int main(int argc, char *argv[]) {
  std::thread camera_thread(UpdateDataFromTimelapseCamera);

  while (true) {
    EstimateFundamentalMatrixBetweenConsecutiveFrames();
    DrawFeatureMatchesOnMonocularStream();
  }

  camera_thread.join();
  return EXIT_SUCCESS;
}