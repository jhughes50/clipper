/*!
* @Author  Jason Hughes
* @Date December 2025
*
* @About script to test the processor mixins
*/

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <clipper/processor_mixins.hpp>

TEST(ProcessorMixinsTestSuite, TestResize)
{
    cv::Mat img = cv::imread("test4.png", cv::IMREAD_COLOR);

    Clipper::ProcessorMixins mixins;

    const int h = 352;
    const int w = 352;

    ASSERT_EQ(img.cols, 2048);
    ASSERT_EQ(img.rows, 1536);

    mixins.resizeImage(img, h, w);

    ASSERT_EQ(img.cols, w);
    ASSERT_EQ(img.rows, h);
}

TEST(ProcessorMixinsTestSuite, TestCvToTensor)
{
    cv::Mat img = cv::imread("test4.png", cv::IMREAD_COLOR);

    Clipper::ProcessorMixins mixins;

    mixins.resizeImage(img, 352, 352);

    at::Tensor tensor = mixins.cvToTensor(img);

    ASSERT_EQ(tensor.size(0), 3);
    ASSERT_EQ(tensor.size(1), 352);
    ASSERT_EQ(tensor.size(2), 352);
}
