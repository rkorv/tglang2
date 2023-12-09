import string

with open("./vocabulary.txt", "r") as f:
    vocab_list = f.read().split("\n")

vocab_list = [word for word in vocab_list if word]

vocab_list += (
    [" ", "\n", "\t", "\t\t", "\t\t\t", "  ", "    ", "        ", "            "]
    + list(string.punctuation)
    + list(string.digits)
    + list(string.ascii_lowercase)
)
letters_pose = len(vocab_list) - len(string.ascii_lowercase + string.digits)
vocab_list += ["<PAD>", "<UNK>"]

unk_idx = vocab_list.index("<UNK>")
pad_idx = vocab_list.index("<PAD>")

# spaces_range = [vocab_list.index(" "), vocab_list.index("            ")]
# max_line_len = 10
max_size = 4096
max_lines_num = 256

vocab_dict = {c: i for i, c in enumerate(vocab_list)}
max_word_len = max([len(word) for word in vocab_list])

naming_types = ["unknown", "lowercase", "uppercase"]
