#include "opencv2/features2d.hpp"
#include <Camera.h>
#include <iostream>
#include <string>
#include <vector>

void viewCamera() {
  vo::common::Camera testcam;
  while (true) {
    const auto new_frame = testcam.Capture();
    if (new_frame.empty()) {
      break;
    }
    const auto window_name{"this is you, smile! :)"};
    cv::imshow(window_name, new_frame);
    const auto escape_key = 27;
    const auto key_press = cv::waitKey(10);
    if (key_press == escape_key) {
      break;
    }
  }
}

void DrawFastCornersOnWebcamStream() {
  vo::common::Camera testcam;
  const auto detector = cv::ORB::create();

  while (true) {
    const cv::Mat new_frame = testcam.Capture();
    if (new_frame.empty()) {
      break;
    }

    std::vector<cv::KeyPoint> keypointsD;
    std::vector<cv::Mat> descriptor;

    detector->detect(new_frame, keypointsD, cv::Mat());
    drawKeypoints(new_frame, keypointsD, new_frame);

    const auto window_name{"FAST corners"};
    cv::imshow(window_name, new_frame);

    const auto escape_key = 27;
    const auto key_press = cv::waitKey(10);
    if (key_press == escape_key) {
      break;
    }
  }
}

int main(int argc, char *argv[]) {
  // viewCamera();
  DrawFastCornersOnWebcamStream();
  return EXIT_SUCCESS;
}
