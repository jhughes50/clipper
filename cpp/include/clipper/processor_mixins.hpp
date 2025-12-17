/*!
* @Author Jason Hughes
* @Date December 2025
*
* @About Image processing basics
*/

#pragma once

#include <opencv2/opencv.hpp>
#include <torch/script.h>

namespace Clipper
{
class ProcessorMixins
{
    public:
        ProcessorMixins() = default;
    
        void normalizeImage(cv::Mat& img, const std::vector<float> mean, const std::vector<float> std);
        void resizeImage(cv::Mat& img, const int h, const int w);

        at::Tensor cvToTensor(const cv::Mat& mat) const noexcept;
        cv::Mat tensorToCv(at::Tensor& tensor) const noexcept;
};
} // namespace Clipper
