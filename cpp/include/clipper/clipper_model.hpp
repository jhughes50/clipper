/*!
* @Author Jason Hughes
* @Date Decemeber 2025 
*
* @About Inference for the CLIPSeg or CLIPper model
*/
#include <string>
#include <torch/script.h>

namespace Clipper
{

strct ClipperModelOutput
{
    at::Tensor logits;
    at::Tensor pooled;
};

class ClipperModelBase
{
    public:
        ClipperModelBase() = default;
        ClipperModelBase(const std::string& model_path, const std::string& proj_path);

        virtual at::Tensor operator()(at::Tensor tensor);

    protected:
        torch::jit::script::Module model_;
        torch::jit::script::Module projection_;
};

class ClipperImageModel : public ClipperModelBase
{
    public:
        using ClipperModelBase::ClipperModelBase;
        at::Tensor operator()(at::Tensor image) override;
};

class ClipperTextModel : public ClipperModelBase
{
    public: 
        using ClipperModelBase::ClipperModelBase;
        at::Tensor operator()(at::Tensor tokens, at::Tensor masks) override;
};

class ClipperDecoderModel
{
    public:
        ClipperDecoderModel() = default;
        ClipperDecoderModel(const std::string& model_path);
        
        at::Tensor operator()(at::Tensor img_embedding, at::Tensor text_embedding);

    private:
        torch::jit::script::Module model_;
};

class ClipperModel
{
    public:
        ClipperModel() = default;
        ClipperModel(const std::string model_path);
        
        ClipperModelOutput operator()(ClipperModelInput inputs);

    private:
        ClipperImageModel image_encoder_;
        ClipperTextModel text_encoder_;
        ClipperDecoderModel decoder_;
};


} // namespace Clipper
