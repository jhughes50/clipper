/*!
* @Author Jason Hughes
* @Date December 2025
*
* @About test the image normalization
*/

#include "clipper/processor_mixins.hpp"

int main()
{
    cv::Mat img = cv::imread("test4.png", cv::IMREAD_COLOR);
    Clipper::ProcessorMixins mixins;

    const std::vector<float> mean = {0.481, 0.457, 0.408};
    const std::vector<float> std = {0.269, 0.261, 0.276};

    mixins.resizeImage(img,352,352);
    mixins.normalizeImage(img, mean, std);

    cv::imshow("img", img);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}   
