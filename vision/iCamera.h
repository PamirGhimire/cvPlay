#pragma once

#include "opencv2/opencv.hpp"

namespace cvp {
namespace vision {

class iCamera {
public:
  iCamera() = default;
  virtual cv::Mat Capture() = 0;
  // todo :  virtual void Calibrate() = 0;
};

} // namespace vision
} // namespace cvp