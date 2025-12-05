/*
* @Author Jason Hughes 
* @Date December 2025
*
* @About prepare the image for CLIPseg model
*/

#pragma once

#include <yaml-cpp/yaml.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include <clipper/processor_mixins.hpp>
#include <clipper/tokenizer.hpp>

namespace Clipper 
{

struct CLIPParameters
{   
    CLIPParameters() = default;
    CLIPParameters(const std::string& path);
    static CLIPParameters Load(const std::string& path);

    std::vector<float> mean;
    std::vector<float> std;
    int height;
    int width;
};

struct CLIPInputs
{
    CLIPInputs() = default;
    CLIPInputs(at::Tensor img, std::vector<at::Tensor> tokens, std::vector<at::Tensor> masks);

    static CLIPInputs InitFromText(std::vector<at::Tensor> tokens, std::vector<at::Tensor> masks);
    static CLIPInputs InitFromImage(at::Tensor img);
    
    // image as a tensor
    at::Tensor image;
    // token ids from embeddings
    std::vector<at::Tensor> tokens;
    // attentions mask
    std::vector<at::Tensor> masks;
};

class CLIPProcessor : public ProcessorMixins
{
    public:
        CLIPProcessor() = default;
        CLIPProcessor(const std::string& params_path, const std::string& merges_path, const std::string& vocab_path); 

        CLIPInputs process(cv::Mat image, std::vector<std::string> text);

    private:
        // downres image, normalize convert to tensor
        at::Tensor processImage(cv::Mat& img);
        CLIPInputs processText(std::vector<std::string>& text);

        CLIPParameters params_;
        CLIPTokenizer tokenizer_;
};
} // namespace Clipper
