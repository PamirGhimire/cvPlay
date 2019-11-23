#pragma once
#include <iCamera.h>

namespace vo
{
namespace common
{

constexpr int kVideoCaptureCamera= 0; //built-in webcam

class Camera : public iCamera
{
public:
    Camera();
    cv::Mat Capture() override;

private:
    cv::VideoCapture webcam_;
};

} //namespace camera ends
} //namespace vo ends