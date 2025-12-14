/*!
* @Author Jason Hughes
* @Date December 2025
*
* @About the CLIPper model
*/
#include <chrono>
#include "clipper/clipper_model.hpp"

using namespace Clipper;

ClipperModelBase::ClipperModelBase(const std::string& model_path, const std::string& proj_path, ClipperModelType type)
{
    if (torch::cuda::is_available()) {
        device_ = std::make_unique<c10::Device>(at::kCUDA);
        std::cout << "Using device: Cuda" << std::endl;
    }
    else {
        device_ = std::make_unique<c10::Device>(at::kCPU);
        std::cout << "Using deviceL CPU" << std::endl;
    }
    model_ = torch::jit::load(model_path);
    model_.to(*device_);
    model_.eval();
    projection_ = torch::jit::load(proj_path);
    projection_.to(*device_);
    projection_.eval();
}

ClipperModelBase::ClipperModelBase(const std::string& model_path, ClipperModelType type)
{
    if (torch::cuda::is_available()) {
        device_ = std::make_unique<c10::Device>(at::kCUDA);
        std::cout << "Using device: Cuda" << std::endl;
    }

    model_ = torch::jit::load(model_path, *device_);
    model_.eval();
}

c10::Device ClipperModelBase::getDevice() const
{
    return *device_;
}

ClipperImageModelOutput ClipperImageModel::operator()(at::Tensor& image)
{
    torch::IValue output = model_({image.to(*device_)});

    // 0 tensor is the pool output
    c10::intrusive_ptr<c10::ivalue::Tuple> output_tuple = output.toTuple();
    at::Tensor pooled = output_tuple->elements()[0].toTensor();
    // 1 tensor is the embedding that needs to be projected
    at::Tensor embedding = output_tuple->elements()[1].toTensor();
    // 2 is tuple of intermediate embeddings 
    c10::intrusive_ptr<c10::ivalue::Tuple> inner_tuple = output_tuple->elements()[2].toTuple();
    std::vector<torch::IValue> inner_elements = inner_tuple->elements();
    
    std::vector<at::Tensor> activations; 
    for (const int layer : embedding_layers_) {
        activations.push_back(inner_elements[layer+1].toTensor());
    }

    torch::IValue proj_output = projection_({embedding});
    at::Tensor final_embedding = proj_output.toTensor();

    ClipperImageModelOutput results{final_embedding, activations};

    return results;
}

at::Tensor ClipperTextModel::operator()(at::Tensor& tokens, at::Tensor& masks)
{
    // forward text pass
    torch::IValue output = model_({tokens.to(*device_), masks.to(*device_)});

    c10::intrusive_ptr<c10::ivalue::Tuple> output_tuple = output.toTuple();
    at::Tensor embedding = output_tuple->elements()[1].toTensor();

    torch::IValue proj_output = projection_({embedding});
    at::Tensor final_embedding = proj_output.toTensor();

    return final_embedding;
}

void ClipperTextModel::setText(std::vector<at::Tensor>& tokens, std::vector<at::Tensor>& masks)
{
    size_t input_size = tokens.size();
    text_embeddings_.clear();

    for (size_t i = 0; i < input_size; ++i) {
        at::Tensor text_embedding = this->operator()(tokens[i], masks[i]);
        text_embeddings_.push_back(text_embedding.detach().cpu().clone());
    }
    
    text_set_ = true;
}

bool ClipperTextModel::isTextSet() const
{
    return text_set_;
}

at::Tensor ClipperTextModel::getTextEmbedding(const size_t idx) const
{
    return text_embeddings_[idx];
}

at::Tensor ClipperDecoderModel::operator()(std::vector<at::Tensor>& img_embeddings, at::Tensor& text_embedding)
{
    c10::List<at::Tensor> activations(img_embeddings);
    torch::IValue output = model_({activations, text_embedding});
    
    c10::intrusive_ptr<c10::ivalue::Tuple> output_tuple = output.toTuple();
    at::Tensor logits = output_tuple->elements()[0].toTensor();

    return logits;
}

ClipperModel::ClipperModel(const std::string& model_dir)
{
    image_encoder_ = ClipperImageModel(model_dir+"/clip-vision-model-traced.pt", 
                                       model_dir+"/clip-vision-projection-traced.pt",
                                       ClipperModelType::IMGENCODER);
    text_encoder_ = ClipperTextModel(model_dir+"/clip-text-model-traced.pt", 
                                     model_dir+"/clip-text-projection-traced.pt",
                                     ClipperModelType::TXTENCODER);

    decoder_ = ClipperDecoderModel(model_dir+"/clip-decoder-traced.pt", ClipperModelType::DECODER);
    
    if (torch::cuda::is_available()) {
        device_ = std::make_unique<c10::Device>(at::kCUDA);
        std::cout << "Using device: Cuda" << std::endl;
    }
    else {
        device_ = std::make_unique<c10::Device>(at::kCPU);
        std::cout << "Using deviceL CPU" << std::endl;
    }
}

void ClipperModel::setText(std::vector<at::Tensor>& tokens, std::vector<at::Tensor>& masks)
{
    text_encoder_.setText(tokens, masks);
}

ClipperModelOutput ClipperModel::operator()(ClipperModelInputs inputs)
{
    ClipperImageModelOutput image_output = image_encoder_(inputs.image);
    
    size_t input_size = inputs.getSize();
    std::vector<at::Tensor> logits;

    for (size_t i = 0; i < input_size; ++i) {
        at::Tensor raw_logits;
        if (!text_encoder_.isTextSet()) {
            at::Tensor text_embedding = text_encoder_(inputs.tokens[i], inputs.masks[i]);
            raw_logits = decoder_(image_output.activations, text_embedding);
        }
        else {
            at::Tensor text_embedding = text_encoder_.getTextEmbedding(i);
            at::Tensor text_embedding_cuda = text_embedding.to(*device_);
            raw_logits = decoder_(image_output.activations, text_embedding_cuda);
        }
        logits.push_back(raw_logits.squeeze_(0));
    }
    std::cout << "logits size: " << logits[0].sizes() << std::endl; 
    ClipperModelOutput output{logits, image_output.activations};
    
    return output;
}   





