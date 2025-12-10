/*!
* @Author Jason Hughes
* @Date December 2025
*
* @About Tokenizer for CLIP
*/

#include "clipper/tokenizer.hpp"

using namespace Clipper;

CLIPTokenizer::CLIPTokenizer(const std::string merges_path, const std::string vocab_path)
{
    merges_ = loadMerges(merges_path);
    vocab_ = loadVocab(vocab_path);
}

std::vector<std::pair<std::string, std::string>> CLIPTokenizer::getMerges() const
{
    return merges_;
}

std::unordered_map<std::string, int> CLIPTokenizer::getVocab() const
{
    return vocab_;
}

std::vector<std::pair<std::string, std::string>> CLIPTokenizer::loadMerges(const std::string& path)
{
    std::vector<std::pair<std::string, std::string>> merges;
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open merges.text file");
    }

    std::string line;
    int idx = 0;
    while (std::getline(file, line)) {
        if (line.empty() || idx == 0) {
            idx += 1;
            continue;
        }

        size_t space_pos = line.find(' ');
        if (space_pos == std::string::npos) continue;

        std::string first = line.substr(0, space_pos);
        std::string second = line.substr(space_pos + 1);

        merges.push_back({first, second});
        idx += 1;
    }

    file.close();
    return merges;
}

std::unordered_map<std::string, int> CLIPTokenizer::loadVocab(const std::string& path)
{
    std::unordered_map<std::string, int> vocab;

    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open vocab.json file");
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errs;

    if (!Json::parseFromStream(builder, file, &root, &errs)) {
        file.close();
        throw std::runtime_error("Failed to parse vocab.json: " + errs);
    }

    file.close();

    Json::Value::Members members = root.getMemberNames();
    for (size_t i = 0; i < members.size(); ++i) {
        std::string token = members[i];
        vocab[token] = root[token].asInt();
    }

    return vocab;
}

std::vector<int> CLIPTokenizer::tokenize(const std::string& text)
{
    std::string processed_text = text;
    std::transform(processed_text.begin(), processed_text.end(), processed_text.begin(), ::tolower);

    std::vector<std::string> words;
    std::istringstream iss(processed_text);
    std::string word;
    while(iss >> word) {
        words.push_back(word);
    }

    std::vector<std::string> all_tokens;

    for (const std::string& w : words) {
        std::vector<std::string> word_tokens;
        for (size_t i = 0; i < w.length(); ++i) {
            word_tokens.push_back(std::string(1, w[i]));
        }

        if (!word_tokens.empty()) {
            word_tokens.back() += "</w>";
        }

        while (word_tokens.size() > 1) {
            int best_merge_idx = -1;
            size_t best_pos = 0;

            for (size_t i = 0; i < word_tokens.size()-1; ++i) {
                std::string pair_key = word_tokens[i] + " " + word_tokens[i+1];

                for (size_t merge_idx = 0; merge_idx < merges_.size(); ++merge_idx) {
                    if (merges_[merge_idx].first == word_tokens[i] && merges_[merge_idx].second == word_tokens[i+1]) {
                        if (best_merge_idx == -1 || merge_idx < best_merge_idx) {
                            best_merge_idx = merge_idx;
                            best_pos = i;
                        }
                        break;
                    }
                }
            }

            if (best_merge_idx == -1) break;

            std::string merged = word_tokens[best_pos] + word_tokens[best_pos+1];
            std::vector<std::string> new_tokens;

            for (size_t i = 0; i < word_tokens.size(); ++i) {
                if (i == best_pos) {
                    new_tokens.push_back(merged);
                } else if (i == best_pos + 1) {
                    continue;
                } else {
                    new_tokens.push_back(word_tokens[i]);
                }
            }
            word_tokens = new_tokens;
        }

        for (const std::string& token : word_tokens) {
            all_tokens.push_back(token);
        }
    }

    std::vector<int> token_ids;
    token_ids.push_back(49406);

    for (const std::string& token : all_tokens) {
        if (vocab_.find(token) != vocab_.end()) {
            token_ids.push_back(vocab_.at(token));
        } else {
            token_ids.push_back(49407);
        }
    }
    token_ids.push_back(49407);

    return token_ids;
}

int CLIPTokenizer::getPaddingToken() const
{
    return 49407;
}
