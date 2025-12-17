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
    tokenizer_ = CLIPTokenizer(merges_path, vocab_path);
}

cv::Mat ClipperProcessor::postProcess(at::Tensor& logits)
{
    cv::Mat heatmap = tensorToCv(logits);
    if (!preprocessed_) {
        LOG(INFO) << "process not called yet, unable to resize to original size";
        return heatmap;
    } else {
        cv::resize(heatmap, heatmap, size_);
    }
    return heatmap;
}

ClipperModelInputs ClipperProcessor::process(cv::Mat image, std::vector<std::string> text)
{
    size_ = image.size();
    preprocessed_ = true;

    ClipperModelInputs inputs= processText(text);
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

ClipperModelInputs ClipperProcessor::processText(std::vector<std::string>& text)
{
    std::vector<at::Tensor> tokens;
    std::vector<at::Tensor> masks;
   
    for (const std::string& t : text) {
        std::vector<int> token_ids = tokenizer_.tokenize(t);
        std::vector<int> attn_mask(token_ids.size(), 1);

        if (token_ids.size() > params_.padding) {
            throw std::runtime_error("Your text input is too long. If you want long inputs adjust the max_length parameter in the python processor used in the compiling script and then adjust the padding parameter in clipper.yaml. You could also tell Jason to write better code, but he probably wont listen.");
        }

        while (token_ids.size() < params_.padding) {
            token_ids.push_back(tokenizer_.getPaddingToken());
            attn_mask.push_back(0);
        }
        at::Tensor token_tensor = torch::tensor(token_ids);
        tokens.push_back(token_tensor.unsqueeze_(0));
        //std::cout << token_tensor.sizes() << std::endl; 
        at::Tensor mask = torch::tensor(attn_mask);
        //std::cout << mask.sizes() << std::endl;
        masks.push_back(mask.unsqueeze_(0));
    }

    ClipperModelInputs inputs = ClipperModelInputs::InitFromText(tokens, masks);
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
        
        padding = config["processor"]["text"]["padding"].as<int>();
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

ClipperModelInputs::ClipperModelInputs(at::Tensor img, std::vector<at::Tensor> tkns, std::vector<at::Tensor> msks)
{
    image = img;
    tokens = tkns;
    masks = msks;
}

ClipperModelInputs ClipperModelInputs::InitFromText(std::vector<at::Tensor> tkns, std::vector<at::Tensor> msks)
{
    ClipperModelInputs inputs;
    inputs.tokens = tkns;
    inputs.masks = msks;

    return inputs;
}

ClipperModelInputs ClipperModelInputs::InitFromImage(at::Tensor img)
{
    ClipperModelInputs inputs;
    inputs.image = img;

    return inputs;
}

size_t ClipperModelInputs::getSize() const
{
    assert(tokens.size() == masks.size() && "Tokens and Mask sizes do not match");
    return tokens.size();
}
