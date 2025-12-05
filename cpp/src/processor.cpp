/*!
* @Author Jason Hughes
* @Date December 2025
*
* @About preprocess the intputs
*/

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "clipper/processor.hpp"

using namespace Clipper;

CLIPProcessor::CLIPProcessor(const std::string& params_path, const std::string& merges_path, const std::string& vocab_path) : ProcessorMixins()
{
    params_ = CLIPParameters::Load(params_path);
    tokenizer_ = CLIPTokenizer(merges_path, vocab_path);
}

CLIPInputs CLIPProcessor::process(cv::Mat image, std::vector<std::string> text)
{
    CLIPInputs inputs= processText(text);
    inputs.image = processImage(image);

    return inputs;
}

at::Tensor CLIPProcessor::processImage(cv::Mat& img)
{
    resizeImage(img, params_.height, params_.width);
    normalizeImage(img, params_.mean, params_.std);
    at::Tensor img_tensor = cvToTensor(img);
    img_tensor.unsqueeze_(0); // 1,C,H,W

    return img_tensor;
}

CLIPInputs CLIPProcessor::processText(std::vector<std::string>& text)
{
    std::vector<at::Tensor> tokens;
    std::vector<at::Tensor> masks;
   
    for (const std::string& t : text) {
        std::vector<int> token_ids = tokenizer_.tokenize(t);
        at::Tensor token_tensor = torch::tensor(token_ids);
        tokens.push_back(token_tensor);
        
        at::Tensor mask = torch::ones({token_ids.size()});
        masks.push_back(mask);
    }

    CLIPInputs inputs = CLIPInputs::InitFromText(tokens, masks);
    return inputs;
}

CLIPParameters::CLIPParameters(const std::string& path)
{
    try {
        YAML::Node config = YAML::LoadFile(path);

        mean = config["processor"]["mean"].as<std::vector<float>>();
        std = config["processor"]["std"].as<std::vector<float>>();

        height = config["processor"]["size"]["height"].as<int>();
        width = config["processor"]["size"]["width"].as<int>();

    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error loading YAML File at : " + path + " : " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error parsing YAML configuration at: " + path + " : " + std::string(e.what()));
    }  
}

CLIPParameters CLIPParameters::Load(const std::string& path)
{
    CLIPParameters params(path);
    return params;
}

CLIPInputs::CLIPInputs(at::Tensor img, std::vector<at::Tensor> tkns, std::vector<at::Tensor> msks)
{
    image = img;
    tokens = tkns;
    masks = msks;
}

CLIPInputs CLIPInputs::InitFromText(std::vector<at::Tensor> tkns, std::vector<at::Tensor> msks)
{
    CLIPInputs inputs;
    inputs.tokens = tkns;
    inputs.masks = msks;

    return inputs;
}

CLIPInputs CLIPInputs::InitFromImage(at::Tensor img)
{
    CLIPInputs inputs;
    inputs.image = img;

    return inputs;
}
