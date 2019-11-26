#include "Camera.h"

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

int main(int argc, char *argv[]) {
  viewCamera();
  return EXIT_SUCCESS;
}