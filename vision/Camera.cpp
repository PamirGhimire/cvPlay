#include <Camera.h>

namespace cvp {
namespace vision {

Camera::Camera() { webcam_.open(kVideoCaptureCamera); }

cv::Mat Camera::Capture() {
  cv::Mat new_frame;
  webcam_.read(new_frame);

  return new_frame;
}

} // namespace vision
} // namespace cvp