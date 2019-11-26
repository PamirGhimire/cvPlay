#include <Camera.h>

namespace vo {
namespace common {

Camera::Camera() { webcam_.open(kVideoCaptureCamera); }

cv::Mat Camera::Capture() {
  cv::Mat new_frame;
  webcam_.read(new_frame);

  return new_frame;
}

} // namespace common
} // namespace vo