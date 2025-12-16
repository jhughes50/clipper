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
#include <glog/logging.h>

#include "clipper/processor.hpp"

namespace Clipper
{

enum ClipperModelType
{
    IMGENCODER,
    TXTENCODER,
    DECODER
};

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
        ClipperModelBase(const std::string& model_path, 
                         const std::string& proj_path,
                         ClipperModelType type);
        ClipperModelBase(const std::string& model_path,
                         ClipperModelType type);

        c10::Device getDevice() const;

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
        
        void setText(std::vector<at::Tensor>& tokens, std::vector<at::Tensor>& masks);

        bool isTextSet() const;
        at::Tensor getTextEmbedding(const size_t idx) const;    

    private:
        std::vector<at::Tensor> text_embeddings_;
        bool text_set_{false};
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
        
        // inference all at once
        ClipperModelOutput operator()(ClipperModelInputs inputs);

        // for more control over inference
        void setText(std::vector<at::Tensor>& tokens, std::vector<at::Tensor>& masks);
        ClipperImageModelOutput setImage(at::Tensor& image);
        at::Tensor inference(std::vector<at::Tensor>& activations, at::Tensor& token, at::Tensor& mask);

    private:
        ClipperImageModel image_encoder_;
        ClipperTextModel text_encoder_;
        ClipperDecoderModel decoder_;
        
        std::unique_ptr<c10::Device> device_;
};


} // namespace Clipper
