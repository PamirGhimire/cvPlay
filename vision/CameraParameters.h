#pragma once

namespace cvp {
namespace vision {

constexpr float kFocalLength = 1000;

struct CameraIntrinsics {
  float focal_x_{kFocalLength};
  float focal_y_{kFocalLength};
  float skew_{0};
  float optical_center_x_{0};
  float optical_center_y_{0};
};

} // namespace vision
} // namespace cvp