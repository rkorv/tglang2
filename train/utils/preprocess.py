import re
from typing import List
from enum import Enum

from utils import vocab


class GroupType(Enum):
    NONE = 0
    QUOTE = 1
    SQUARE_BRACKET = 2
    PARENTHESIS = 3
    CURLY_BRACKET = 4


def remove_minimum_leading(raw):
    lines = raw.splitlines()

    def count_leading(s):
        count = 0
        for char in s:
            if char in " \t":
                count += 1
            else:
                break
        return count

    leadings = [count_leading(line) for line in lines if line.strip()]
    if not leadings:
        return raw
    min_leading = min(leadings)
    result = [line[min_leading:].rstrip() for line in lines if line.strip()]
    if not result:
        return raw
    return "\n".join(result)


def analyze_string(s, max_lines=256):
    line_numbers = []
    group_type = []

    current_line = 0
    group_stack = []

    for char in s:
        line_numbers.append(current_line % max_lines)

        if char == ['"', "'"]:
            if group_stack and group_stack[-1] == GroupType.QUOTE:
                group_stack.pop()
            else:
                group_stack.append(GroupType.QUOTE)
        elif char == "[":
            group_stack.append(GroupType.SQUARE_BRACKET)
        elif char == "]":
            if group_stack and group_stack[-1] == GroupType.SQUARE_BRACKET:
                group_stack.pop()
        elif char == "(":
            group_stack.append(GroupType.PARENTHESIS)
        elif char == ")":
            if group_stack and group_stack[-1] == GroupType.PARENTHESIS:
                group_stack.pop()
        elif char == "{":
            group_stack.append(GroupType.CURLY_BRACKET)
        elif char == "}":
            if group_stack and group_stack[-1] == GroupType.CURLY_BRACKET:
                group_stack.pop()

        group_type.append(group_stack[-1].value if group_stack else GroupType.NONE.value)

        if char == "\n":
            current_line += 1

    return line_numbers, group_type


def encode_text(text):
    text = remove_minimum_leading(text)
    s_line_numbers, s_group_types = analyze_string(text)

    line_numbers = []
    group_types = []
    encoded_text = []
    naming_types = []
    positions = []

    i = 0

    lower_text = text.lower()
    while i < len(text):
        for j in range(min(i + vocab.max_word_len, len(text)), i, -1):
            word = lower_text[i:j]
            if word in vocab.vocab_dict:
                line_numbers.append(s_line_numbers[i])
                group_types.append(s_group_types[i])

                original_word = text[i:j]
                if original_word.islower():
                    naming_types.append(1)
                elif original_word.isupper():
                    naming_types.append(2)
                else:
                    naming_types.append(0)

                encoded_text.append(vocab.vocab_dict[word])

                positions.append(i)
                i = j
                break
        else:
            encoded_text.append(vocab.vocab_dict["<UNK>"])
            naming_types.append(0)
            line_numbers.append(s_line_numbers[i])
            group_types.append(s_group_types[i])
            positions.append(i)
            i += 1

    return encoded_text, naming_types, group_types, line_numbers, positions


def decode_text(encoded_text):
    return "".join([vocab.vocab_list[c] for c in encoded_text])
