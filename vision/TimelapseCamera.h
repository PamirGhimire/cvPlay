#pragma once
#include <Camera.h>
#include <array>
#include <deque>

namespace cvp {
namespace vision {

using uint = unsigned int;

struct TimeSeparatedFrames {
  TimeSeparatedFrames(cv::Mat current_frame, cv::Mat delayed_frame,
                      bool is_valid)
      : current_frame_{current_frame}, delayed_frame_(delayed_frame),
        is_valid_(is_valid) {}

  TimeSeparatedFrames() = default;

  bool IsValid() const {
    if (is_valid_) {
      return true;
    }
    return false;
  }

  cv::Mat current_frame_{};
  cv::Mat delayed_frame_{};
  bool is_valid_{false};
};

class TimelapseCamera : public Camera {
public:
  TimelapseCamera(uint number_of_frames_to_skip_between_separated_frames);
  TimeSeparatedFrames GetTimeSeparatedFrames();
  bool IsCameraReady() const;
  void Update();

private:
  uint number_of_frames_to_skip_between_separated_frames_{0};
  int number_of_frames_skipped_{-2};
  std::deque<cv::Mat> frames_buffer_{};
};

} // namespace vision
} // namespace cvp