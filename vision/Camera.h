#pragma once
#include <iCamera.h>

namespace cvp {
namespace vision {

constexpr int kVideoCaptureCamera = 0; // built-in webcam

class Camera : public iCamera {
public:
  Camera();
  cv::Mat Capture() override;

private:
  cv::VideoCapture webcam_;
};

} // namespace vision
} // namespace cvp