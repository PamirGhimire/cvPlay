#pragma once

namespace common
{
namespace camera
{

constexpr float32_t kFocalLength = 1000;

struct CameraIntrinsics
{
    float32_t focal_x_{kFocalLength};
    float32_t focal_y_{kFocalLength};
    float32_t skew_{0};
    float32_t optical_center_x_{0};
    float32_t optical_center_y_{0};
};

}//namespace camera ends
}//namespace common ends