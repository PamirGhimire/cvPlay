#include "FeatureMatcher.h"
#include "TimelapseCamera.h"

void ShowMatchesBetweenTimeSeparatedFrames(
    const cvp::vision::TimeSeparatedFrames &time_separated_frames) {
  using namespace cvp::vision;
  cv::Mat image_left = time_separated_frames.current_frame_;
  cv::Mat image_right = time_separated_frames.delayed_frame_;

  visual_features::FeatureMatcher feature_matcher;
  const auto feature_correspondences_left_right =
      feature_matcher.FindCorrespondencesBetweenTwoImages(image_left,
                                                          image_right);
  const auto feature_matches_left_right =
      feature_correspondences_left_right.matches_in_dest_right_for_left_;

  cv::Mat image_showing_matches;

  if (!feature_matches_left_right.empty()) {
    const auto keypoints_in_left_image =
        feature_correspondences_left_right.keypoints_left_image_;
    const auto keypoints_in_right_image =
        feature_correspondences_left_right.keypoints_right_image_;

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
    const auto time_separated_frames =
        timelapse_camera.GetTimeSeparatedFrames();

    if (time_separated_frames.IsValid())
      ShowMatchesBetweenTimeSeparatedFrames(time_separated_frames);

    if (UserPressedEscapeKey()) {
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  DrawFeatureMatchesOnMonocularStream();
  return EXIT_SUCCESS;
}