#pragma once

#include <cstring>
#include <vector>
#include <algorithm>
#include <sstream>

#include <model_meta.hpp>

class TglangPreprocessor
{

public:
    void preprocess(
        const char *text,
        std::vector<det_int_t> &encoded_text,
        std::vector<det_int_t> &naming_types,
        std::vector<det_int_t> &group_types,
        std::vector<det_int_t> &lines_num,
        std::vector<det_int_t> &positions_ids)
    {
        std::string text_str = format_text(text);

        if (text_str.empty())
            return;

        encode_text(text_str, encoded_text, naming_types, group_types, lines_num, positions_ids);

        if (encoded_text.size() > MODEL_MAX_INPUT)
        {
            encoded_text.resize(MODEL_MAX_INPUT);
            naming_types.resize(MODEL_MAX_INPUT);
            group_types.resize(MODEL_MAX_INPUT);
            lines_num.resize(MODEL_MAX_INPUT);
            positions_ids.resize(MODEL_MAX_INPUT);
        }
    }

private:
    enum GroupType
    {
        NONE = 0,
        QUOTE = 1,
        SQUARE_BRACKET = 2,
        PARENTHESIS = 3,
        CURLY_BRACKET = 4
    };

    const size_t TGLANG_MAX_STRING_LEN = 4096;

    size_t count_leading_spaces(const std::string &line) const
    {
        return line.find_first_not_of(" \t");
    }

    void rtrim(std::string &line) const
    {
        line.erase(line.find_last_not_of(" \t") + 1);
    }

    det_int_t get_naming_type(const std::string_view &word) const
    {
        det_int_t type = 0;
        for (auto c : word)
        {
            if (islower(c))
            {
                if (type == 2)
                    return 0;
                type = 1;
            }
            else if (isupper(c))
            {
                if (type == 1)
                    return 0;
                type = 2;
            }
            else
                return 0;
        }
        return type;
    }

    void analyze_string(
        const std::string &s,
        std::vector<det_int_t> &line_nums,
        std::vector<det_int_t> &group_ids)
    {
        int current_line = 0;
        std::vector<GroupType> group_stack;

        for (char ch : s)
        {
            line_nums.push_back(current_line % MODEL_MAX_LINES_NUM);

            if (ch == '\"' || ch == '\'')
            {
                if (!group_stack.empty() && group_stack.back() == QUOTE)
                {
                    group_stack.pop_back();
                }
                else
                {
                    group_stack.push_back(QUOTE);
                }
            }
            else if (ch == '[')
            {
                group_stack.push_back(SQUARE_BRACKET);
            }
            else if (ch == ']')
            {
                if (!group_stack.empty() && group_stack.back() == SQUARE_BRACKET)
                {
                    group_stack.pop_back();
                }
            }
            else if (ch == '(')
            {
                group_stack.push_back(PARENTHESIS);
            }
            else if (ch == ')')
            {
                if (!group_stack.empty() && group_stack.back() == PARENTHESIS)
                {
                    group_stack.pop_back();
                }
            }
            else if (ch == '{')
            {
                group_stack.push_back(CURLY_BRACKET);
            }
            else if (ch == '}')
            {
                if (!group_stack.empty() && group_stack.back() == CURLY_BRACKET)
                {
                    group_stack.pop_back();
                }
            }

            group_ids.push_back(!group_stack.empty() ? group_stack.back() : NONE);

            if (ch == '\n')
                current_line++;
        }
    }

    void encode_text(
        const std::string &inptext,
        std::vector<det_int_t> &encoded_text,
        std::vector<det_int_t> &naming_types,
        std::vector<det_int_t> &group_types,
        std::vector<det_int_t> &lines_num,
        std::vector<det_int_t> &positions_ids)
    {
        std::string text = inptext;
        std::transform(text.begin(), text.end(), text.begin(), ::tolower);

        std::vector<det_int_t> s_lines_num, s_group_types;
        analyze_string(text, s_lines_num, s_group_types);

        encoded_text.reserve(text.size());
        naming_types.reserve(text.size());
        group_types.reserve(text.size());
        lines_num.reserve(text.size());
        positions_ids.reserve(text.size());

        auto i = text.begin();
        while (i != text.end())
        {
            bool found = false;
            auto end = i + std::min(VOCAB_MAX_LEN, static_cast<int>(text.end() - i));
            auto start_pos = std::distance(text.begin(), i);

            for (auto j = end; j != i; --j)
            {
                if (vocab_map.find({i, j}) != vocab_map.end())
                {
                    encoded_text.push_back(vocab_map.at({i, j}));
                    std::string_view original_substring_view(inptext.data() + start_pos, std::distance(i, j));
                    naming_types.push_back(get_naming_type(original_substring_view));
                    positions_ids.push_back(start_pos);

                    group_types.push_back(s_group_types[start_pos]);
                    lines_num.push_back(s_lines_num[start_pos]);

                    i = j;
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                encoded_text.push_back((det_int_t)VOCAB_UNK_ID);
                naming_types.push_back(0);
                positions_ids.push_back(start_pos);
                group_types.push_back(s_group_types[start_pos]);
                lines_num.push_back(s_lines_num[start_pos]);
                ++i;
            }
        }
    }

    /**
     * Convert text to string and preprocess it:
     *  - remove the minimum number of leading spaces
     *  - remove spaces and tabs at the end of each line
     *  - remove empty lines
     */
    std::string format_text(const char *text)
    {
        std::string raw(text, std::min(this->TGLANG_MAX_STRING_LEN, strlen(text)));

        std::istringstream stream(raw);
        std::string line;
        std::vector<std::string> lines;

        while (std::getline(stream, line))
            lines.push_back(line);

        size_t min_leading = SIZE_MAX;
        for (const auto &line : lines)
            if (!line.empty() && line.find_first_not_of(" \t") != std::string::npos)
                min_leading = std::min(min_leading, count_leading_spaces(line));

        std::string result;
        for (size_t i = 0; i < lines.size(); ++i)
        {
            if (!lines[i].empty() && lines[i].find_first_not_of(" \t") != std::string::npos)
            {
                lines[i] = lines[i].substr(min_leading);
                rtrim(lines[i]);
                result += lines[i];
                if (i != lines.size() - 1)
                    result += "\n";
            }
        }

        return result;
    }
};