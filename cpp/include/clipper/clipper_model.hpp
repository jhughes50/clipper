/*!
* @Author Jason Hughes
* @Date Decemeber 2025 
*
* @About Inference for the CLIPSeg or CLIPper model
*/
#pragma once

#include <string>
#include <torch/script.h>
#include <torch/torch.h>

#include "clipper/processor.hpp"

namespace Clipper
{

struct ClipperModelOutput
{
    std::vector<at::Tensor> logits;
    std::vector<at::Tensor> activations;
};

struct ClipperImageModelOutput
{
    at::Tensor embedding;
    std::vector<at::Tensor> activations;
};

class ClipperModelBase
{
    public:
        ClipperModelBase() = default;
        ClipperModelBase(const std::string& model_path, const std::string& proj_path);
        ClipperModelBase(const std::string& model_path);

    protected:
        torch::jit::script::Module model_;
        torch::jit::script::Module projection_;
        std::unique_ptr<c10::Device> device_;
};

class ClipperImageModel : public ClipperModelBase
{
    public:
        using ClipperModelBase::ClipperModelBase;
        ClipperImageModelOutput operator()(at::Tensor& image);

    private:
        int embedding_layers_[3] = {3,6,9};
};

class ClipperTextModel : public ClipperModelBase
{
    public: 
        using ClipperModelBase::ClipperModelBase;
        at::Tensor operator()(at::Tensor& tokens, at::Tensor& masks);
};

class ClipperDecoderModel : public ClipperModelBase
{
    public:
        using ClipperModelBase::ClipperModelBase;
        at::Tensor operator()(std::vector<at::Tensor>& img_embeddings, at::Tensor& text_embedding);
};

class ClipperModel
{
    public:
        ClipperModel() = default;
        ClipperModel(const std::string& model_dir);
        
        ClipperModelOutput operator()(ClipperModelInputs inputs);

        // TODO give direct access to forward pass 

    private:
        ClipperImageModel image_encoder_;
        ClipperTextModel text_encoder_;
        ClipperDecoderModel decoder_;
};


} // namespace Clipper
