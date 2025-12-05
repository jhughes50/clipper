
/*!
* @Author Jason Hughes
* @Date Decemeber 2025
*
* @About test tokenization
*/

#include <gtest/gtest.h>

#include "clipper/tokenizer.hpp"

TEST(TokenizerTestSuite, TestTokenizePavement)
{
    Clipper::CLIPTokenizer tokenizer("../config/merges.txt", "../config/vocab.json");
    
    std::vector<int> tokens = tokenizer.tokenize("pavement");

    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0], 49406); 
    ASSERT_EQ(tokens[1], 27669);
    ASSERT_EQ(tokens[2], 49407);
}

TEST(TokenizerTestSuite, TestTokenizeDog)
{
    Clipper::CLIPTokenizer tokenizer("../config/merges.txt", "../config/vocab.json");
    
    std::vector<int> tokens = tokenizer.tokenize("dog");

    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0], 49406); 
    ASSERT_EQ(tokens[1], 1929);
    ASSERT_EQ(tokens[2], 49407);
}

TEST(TokenizerTestSuite, TestTokenizeRoad)
{
    Clipper::CLIPTokenizer tokenizer("../config/merges.txt", "../config/vocab.json");
    
    std::vector<int> tokens = tokenizer.tokenize("road");

    ASSERT_EQ(tokens.size(), 3);
    ASSERT_EQ(tokens[0], 49406); 
    ASSERT_EQ(tokens[1], 1759);
    ASSERT_EQ(tokens[2], 49407);
}

TEST(TokenizerTestSuite, TestTokenizeHuman)
{
    Clipper::CLIPTokenizer tokenizer("../config/merges.txt", "../config/vocab.json");
    
    std::vector<int> tokens = tokenizer.tokenize("human on a horse");

    ASSERT_EQ(tokens.size(), 6);
    ASSERT_EQ(tokens[0], 49406); 
    ASSERT_EQ(tokens[1], 2751);
    ASSERT_EQ(tokens[2], 525);
    ASSERT_EQ(tokens[3], 320);
    ASSERT_EQ(tokens[4], 4558);
    ASSERT_EQ(tokens[5], 49407);
}
