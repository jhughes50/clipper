/*!
* @Author Jason Hughes
* @Date Decemeber 2025
*
* @About test the merge file loading
*/

#include <gtest/gtest.h>

#include "clipper/tokenizer.hpp"

TEST(TokenizerTestSuite, TestMergeLoad)
{
    Clipper::CLIPTokenizer tokenizer("../config/merges.txt", "../config/vocab.txt");
    std::vector<std::pair<std::string, std::string>> merge = tokenizer.getMerges();

    ASSERT_EQ(merge[0].first, "i");
    ASSERT_EQ(merge[0].second, "n");

    ASSERT_EQ(merge[6].first, "th");
    ASSERT_EQ(merge[6].second, "e</w>");
}
