/*!
* @Author Jason Hughes
* @Date December 2025
*
* @About Image processing basics
*/

#include <clipper/processor_mixins.hpp>

using namespace Clipper;

void ProcessorMixins::normalizeImage(cv::Mat& img, const std::vector<float> mean, const std::vector<float> std)
{
    // expects image in BGR format
    img.convertTo(img, CV_32F);
    img /= 255.0f;
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    std::vector<cv::Mat> channels;
    cv::split(img, channels);

    for (size_t i = 0; i < channels.size(); ++i) {
        channels[i] = (channels[i] - mean[i]) / std[i];
    }
    // gives image back in RGB format
    cv::merge(channels, img);
}

void ProcessorMixins::resizeImage(cv::Mat& img, const int h, const int w)
{
    cv::Size dims(w, h);
    cv::resize(img, img, dims, cv::INTER_AREA);
}

at::Tensor ProcessorMixins::cvToTensor(const cv::Mat& mat) const noexcept
{
    cv::Mat c_img = mat.isContinuous() ? mat : mat.clone();

    at::Tensor tensor_img = torch::from_blob(c_img.data, {c_img.rows, c_img.cols, 3});
    tensor_img = tensor_img.permute({2, 0, 1});
    tensor_img = tensor_img.to(torch::kFloat32);

    return tensor_img;
}

cv::Mat ProcessorMixins::tensorToCv(at::Tensor& tensor) const noexcept
{
    int rows = tensor.size(0);
    int cols = tensor.size(1);

    cv::Mat mat(rows, cols, CV_32FC1, tensor.data_ptr<float>());

    return mat.clone();
}
