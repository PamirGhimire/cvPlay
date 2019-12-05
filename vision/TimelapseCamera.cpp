#include <TimelapseCamera.h>

namespace cvp {
namespace vision {

TimelapseCamera::TimelapseCamera(
    uint number_of_frames_to_skip_between_separated_frames) {
  number_of_frames_to_skip_between_separated_frames_ =
      number_of_frames_to_skip_between_separated_frames;
}

bool TimelapseCamera::IsCameraReady() const {
  const auto camera_is_ready =
      frames_buffer_.size() ==
      number_of_frames_to_skip_between_separated_frames_ + 2;
  return camera_is_ready;
}

TimeSeparatedFrames TimelapseCamera::GetTimeSeparatedFrames() {
  if (IsCameraReady()) {
    return {frames_buffer_[0], frames_buffer_.back(), true};
  }
  return TimeSeparatedFrames{cv::Mat{}, cv::Mat{}, false};
}

void TimelapseCamera::Update() {
  const auto new_frame = Camera::Capture();
  if (number_of_frames_skipped_ ==
      number_of_frames_to_skip_between_separated_frames_) {
    frames_buffer_.push_front(new_frame);
    frames_buffer_.pop_back();
    return;
  }

  frames_buffer_.push_front(new_frame);
  ++number_of_frames_skipped_;
}

} // namespace vision
} // namespace cvp