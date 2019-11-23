#pragma once

#include "opencv2/opencv.hpp"

namespace vo
{
namespace common
{

class iCamera
{
public:
    iCamera() = default;
    virtual cv::Mat Capture() = 0;
    //virtual void Calibrate() = 0;
};

} //namespace camera ends
} //namespace vo ends   