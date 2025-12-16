/*!
* @Author Jason Hughes
* @Date Decemeber 2025
*
* @About test the model
*/

#include "clipper/clipper_model.hpp"
#include "clipper/processor.hpp"

int main()
{
    Clipper::ClipperModel model("../models");
    Clipper::ClipperProcessor processor("../config/clipper.yaml",
                                        "../config/merges.txt",
                                        "../config/vocab.json");

    cv::Mat img = cv::imread("test4.png", cv::IMREAD_COLOR);
    std::vector<std::string> texts = {"road"};

    Clipper::ClipperModelInputs inputs = processor.process(img, texts);
    Clipper::ClipperImageModelOutput image_output = model.setImage(inputs.image);
    size_t input_size = inputs.getSize();

    std::vector<at::Tensor> outputs;
    for (size_t i = 0; i < input_size; ++i) {
        at::Tensor logits = model.inference(image_output.activations, 
                                            inputs.tokens[i], 
                                            inputs.masks[i]);
        outputs.push_back(logits);
    }

    cv::Mat heatmap = processor.postProcess(outputs[0]);

    cv::imshow("heatmap", heatmap);
    cv::waitKey(0);

    return 0;
}
