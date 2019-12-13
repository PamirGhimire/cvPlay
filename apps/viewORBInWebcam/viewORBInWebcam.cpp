#include "Camera.h"
#include "FeatureExtractor.h"

void DrawFastCornersOnWebcamStream() {
  cvp::vision::Camera testcam;
  cvp::vision::visual_features::FeatureExtractor feature_extractor;

  while (true) {
    const cv::Mat new_frame = testcam.Capture();
    if (new_frame.empty()) {
      break;
    }

    const auto orb_keypoints_and_features =
        feature_extractor.GetORBKeypointsAndDescriptorsFromImage(new_frame);
    const auto orb_keypoints = orb_keypoints_and_features.keypoints_;

    cv::drawKeypoints(new_frame, orb_keypoints, new_frame);

    const auto window_name{"ORB corners"};
    cv::imshow(window_name, new_frame);

    const auto escape_key = 27;
    const auto key_press = cv::waitKey(10);
    if (key_press == escape_key) {
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  DrawFastCornersOnWebcamStream();
  return EXIT_SUCCESS;
}