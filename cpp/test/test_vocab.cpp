/*!
* @Author Jason Hughes
* @Date December 2025
*
* @About test the vocab file
*/

#include <gtest/gtest.h>

#include "clipper/tokenizer.hpp"

TEST(TokenizerTestSuite, TestVocabLoad)
{ 
    Clipper::CLIPTokenizer tokenizer("../config/merges.txt", "../config/vocab.json");
    std::unordered_map<std::string, int> vocab = tokenizer.getVocab();

    ASSERT_EQ(vocab["!"], 0);
}
