/*!
* @Author Jason Hughes
* @Date December 2025 
*
* @About test the preprocessing
*/

#include <gtest/gtest.h>

#include "clipper/processor.hpp"

TEST(ProcessorTestSuite, TestProcessing)
{ 
    cv::Mat img = cv::imread("test4.png", cv::IMREAD_COLOR);
    std::vector<std::string> texts = {"pavement", "roads", "car"};

    std::string path = "../config/";
    std::string merges_path = path + "merges.txt";
    std::string vocab_path = path + "vocab.json";
    std::string params_path = path + "clipper.yaml";

    Clipper::ClipperProcessor processor(params_path, merges_path, vocab_path);

    Clipper::ClipperModelInputs inputs = processor.process(img, texts);
}
