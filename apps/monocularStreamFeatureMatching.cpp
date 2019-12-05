#include "FeatureMatcher.h"
#include "TimelapseCamera.h"

void ShowMatchesBetweenTimeSeparatedFrames(
    const cvp::vision::TimeSeparatedFrames &time_separated_frames) {
  using namespace cvp::vision;
  cv::Mat image_left = time_separated_frames.current_frame_;
  cv::Mat image_right = time_separated_frames.delayed_frame_;

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

    const auto escape_key = 27;
    const auto key_press = cv::waitKey(10);
    if (key_press == escape_key) {
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  DrawFeatureMatchesOnMonocularStream();
  return EXIT_SUCCESS;
}