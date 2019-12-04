#include "FeatureMatcher.h"
#include "TimelapseCamera.h"
#include <thread>

struct AppData {
  vo::common::TimeSeparatedFrames time_separated_frames_;
  bool app_is_alive_{true};
};
static AppData app_data;

void UpdateDataFromTimelapseCamera() {
  using namespace vo::common;
  const auto number_of_frames_to_skip_between_matched_frames = 5;
  TimelapseCamera timelapse_camera(
      number_of_frames_to_skip_between_matched_frames);

  while (app_data.app_is_alive_) {
    timelapse_camera.Update();
    app_data.time_separated_frames_ = timelapse_camera.GetTimeSeparatedFrames();
  }
}

void ShowMatchesBetweenTimeSeparatedFrames() {
  using namespace vo::common;
  cv::Mat image_left = app_data.time_separated_frames_.current_frame_;
  cv::Mat image_right = app_data.time_separated_frames_.delayed_frame_;

  visual_features::FeatureMatcher feature_matcher;
  const auto feature_matches_left_right =
      feature_matcher.MatchKeypointsInImages(image_left, image_right);

  cv::Mat image_showing_matches;

  if (!feature_matches_left_right.empty()) {
    visual_features::FeatureExtractor feature_extractor;
    const auto orb_keypoints_image_left =
        feature_extractor.GetORBKeypointsInImage(image_left);
    const auto orb_keypoints_image_right =
        feature_extractor.GetORBKeypointsInImage(image_right);

    cv::drawMatches(image_left, orb_keypoints_image_left, image_right,
                    orb_keypoints_image_right, feature_matches_left_right,
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
  while (true) {
    if (app_data.time_separated_frames_.IsValid())
      ShowMatchesBetweenTimeSeparatedFrames();

    if (UserPressedEscapeKey()) {
      app_data.app_is_alive_ = false;
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  std::thread camera_thread(UpdateDataFromTimelapseCamera);
  DrawFeatureMatchesOnMonocularStream();

  camera_thread.join();
  return EXIT_SUCCESS;
}