/*
* @Author Jason Hughes 
* @Date December 2025
*
* @About prepare the image for Clipperseg model
*/

#pragma once

#include <cassert>
#include <yaml-cpp/yaml.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <clipper/processor_mixins.hpp>
#include <clipper/tokenizer.hpp>

namespace Clipper 
{

struct ClipperParameters
{   
    ClipperParameters() = default;
    ClipperParameters(const std::string& path);
    static ClipperParameters Load(const std::string& path);

    std::vector<float> mean;
    std::vector<float> std;
    int height;
    int width;
    int padding;
};

struct ClipperModelInputs
{
    ClipperModelInputs() = default;
    ClipperModelInputs(at::Tensor img, std::vector<at::Tensor> tokens, std::vector<at::Tensor> masks);

    static ClipperModelInputs InitFromText(std::vector<at::Tensor> tokens, std::vector<at::Tensor> masks);
    static ClipperModelInputs InitFromImage(at::Tensor img);
    
    // image as a tensor
    at::Tensor image;
    // token ids from embeddings
    std::vector<at::Tensor> tokens;
    // attentions mask
    std::vector<at::Tensor> masks;

    size_t getSize() const;
};

class ClipperProcessor : public ProcessorMixins
{
    public:
        ClipperProcessor() = default;
        ClipperProcessor(const std::string& params_path, const std::string& merges_path, const std::string& vocab_path); 

        ClipperModelInputs process(cv::Mat image, std::vector<std::string> text);

    private:
        // downres image, normalize convert to tensor
        at::Tensor processImage(cv::Mat& img);
        ClipperModelInputs processText(std::vector<std::string>& text);

        ClipperParameters params_;
        CLIPTokenizer tokenizer_;
};
} // namespace Clipper
