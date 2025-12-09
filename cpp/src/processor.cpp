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

ClipperProcessor::ClipperProcessor(const std::string& params_path, const std::string& merges_path, const std::string& vocab_path) : ProcessorMixins()
{
    params_ = ClipperParameters::Load(params_path);
    tokenizer_ = ClipperTokenizer(merges_path, vocab_path);
}

ClipperInputs ClipperProcessor::process(cv::Mat image, std::vector<std::string> text)
{
    ClipperInputs inputs= processText(text);
    inputs.image = processImage(image);

    return inputs;
}

at::Tensor ClipperProcessor::processImage(cv::Mat& img)
{
    resizeImage(img, params_.height, params_.width);
    normalizeImage(img, params_.mean, params_.std);
    at::Tensor img_tensor = cvToTensor(img);
    img_tensor.unsqueeze_(0); // 1,C,H,W

    return img_tensor;
}

ClipperInputs ClipperProcessor::processText(std::vector<std::string>& text)
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

    ClipperInputs inputs = ClipperInputs::InitFromText(tokens, masks);
    return inputs;
}

ClipperParameters::ClipperParameters(const std::string& path)
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

ClipperParameters ClipperParameters::Load(const std::string& path)
{
    ClipperParameters params(path);
    return params;
}

ClipperInputs::ClipperInputs(at::Tensor img, std::vector<at::Tensor> tkns, std::vector<at::Tensor> msks)
{
    image = img;
    tokens = tkns;
    masks = msks;
}

ClipperInputs ClipperInputs::InitFromText(std::vector<at::Tensor> tkns, std::vector<at::Tensor> msks)
{
    ClipperInputs inputs;
    inputs.tokens = tkns;
    inputs.masks = msks;

    return inputs;
}

ClipperInputs ClipperInputs::InitFromImage(at::Tensor img)
{
    ClipperInputs inputs;
    inputs.image = img;

    return inputs;
}
