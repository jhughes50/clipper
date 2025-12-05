/*!
* @Author: Jason Hughes
* @Date: December 2025
*
* @About: a CLIP tokenizer
*/

#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <utility>
#include <unordered_map>
#include <algorithm>
#include <set>

#include <json/json.h>

namespace Clipper
{

class CLIPTokenizer
{
    public:
        CLIPTokenizer() = default;
        CLIPTokenizer(const std::string merge_path, const std::string vocab_path);
    
        std::vector<int> tokenize(const std::string& text);

        std::vector<std::pair<std::string, std::string>> getMerges() const;
        std::unordered_map<std::string, int> getVocab() const;

    private:
        std::vector<std::pair<std::string, std::string>> loadMerges(const std::string& path);
        std::unordered_map<std::string, int> loadVocab(const std::string& path);

        std::vector<std::pair<std::string, std::string>> merges_;
        std::unordered_map<std::string, int> vocab_;
};
} // namespace Clipper
