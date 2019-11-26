#pragma once

namespace common {
namespace camera {

constexpr float kFocalLength = 1000;

struct CameraIntrinsics {
  float focal_x_{kFocalLength};
  float focal_y_{kFocalLength};
  float skew_{0};
  float optical_center_x_{0};
  float optical_center_y_{0};
};

} // namespace camera
} // namespace common