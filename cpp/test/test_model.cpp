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

    Clipper::ClipperModelOutput output = model(inputs);

    cv::Mat heatmap = processor.postProcess(output.logits[0]);

    cv::imshow("heatmap", heatmap);
    cv::waitKey(0);

    return 0;
}
