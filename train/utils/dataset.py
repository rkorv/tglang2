import random
import torch
from torch.utils.data import Dataset

from utils import lang_enum, vocab, helper, preprocess


class CodeDataset(Dataset):
    def __init__(self, df, max_seq_len=4096, max_strlen=4096, aug=True, crop_augs=(1, 60)):
        df.loc[df["language_tag"] == "TGLANG_LANGUAGE_CODE", "language_tag"] = "TGLANG_LANGUAGE_C"
        df["language_id"] = df["language_tag"].map(lang_enum.languages2num)
        self.paths = df["path"].tolist()
        self.labels = df["language_id"].tolist()
        self.source = df["source"].tolist()
        self.max_strlen = max_strlen
        self.max_seq_len = max_seq_len
        self.crop_augs = crop_augs

        self.aug = aug

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def collate_fn(batch):
        (
            inputs,
            naming_types,
            group_types,
            line_ids,
            positions,
            labels,
            attention_masks,
        ) = zip(*batch)
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=vocab.pad_idx)
        positions = torch.nn.utils.rnn.pad_sequence(positions, batch_first=True, padding_value=0).long()
        naming_types = torch.nn.utils.rnn.pad_sequence(naming_types, batch_first=True, padding_value=0).long()
        group_types = torch.nn.utils.rnn.pad_sequence(group_types, batch_first=True, padding_value=0).long()
        line_ids = torch.nn.utils.rnn.pad_sequence(line_ids, batch_first=True, padding_value=0).long()
        attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
        labels = torch.stack(labels).float()
        # return (
        #     inputs,
        #     naming_types,
        #     group_types,
        #     line_ids,
        #     positions,
        #     labels,
        #     attention_masks,
        # )
        return (
            inputs,
            naming_types,
            group_types,
            line_ids,
            positions,
            labels,
            attention_masks,
        )

    def _cut_aug(self, text):
        if len(text) < 30:
            return text

        cut_len = torch.randint(len(text) // 2, len(text), (1,)).item()
        cut_start = torch.randint(0, len(text) - cut_len, (1,)).item()
        cut_end = cut_start + cut_len
        return text[cut_start:cut_end]

    def _remove_random_chars(self, text):
        ration = 0.02
        num_chars_to_remove = int(len(text) * ration)
        chars_to_remove = random.sample(range(len(text)), num_chars_to_remove)
        return "".join([char for idx, char in enumerate(text) if idx not in chars_to_remove])

    def _get_item(self, idx):
        with open(self.paths[idx], "r") as f:
            text = f.read()

        if self.aug:
            text = helper.augment(text, lines_num_range=self.crop_augs)
            if (self.source[idx] == "llama_stories" or self.source[idx] == "stackoverflow") and torch.rand(1) < 0.5:
                text = self._cut_aug(text)

            if torch.rand(1) < 0.1:
                text = self._remove_random_chars(text)

        text = text[: self.max_strlen - 1]

        is_empty = len(text) == 0

        inputs, naming_types, group_types, line_ids, positions = preprocess.encode_text(text)[: self.max_strlen]

        inputs = torch.tensor(inputs, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.long)
        naming_types = torch.tensor(naming_types, dtype=torch.long)
        label = torch.tensor(self.labels[idx] if not is_empty else 0, dtype=torch.long)
        label = torch.nn.functional.one_hot(label, len(lang_enum.languages)).float()
        group_types = torch.tensor(group_types, dtype=torch.long)
        line_ids = torch.tensor(line_ids, dtype=torch.long)

        inputs = inputs[: self.max_seq_len]
        naming_types = naming_types[: self.max_seq_len]
        positions = positions[: self.max_seq_len]
        attention_masks = torch.ones_like(inputs)
        line_ids = line_ids[: self.max_seq_len]
        group_types = group_types[: self.max_seq_len]

        return inputs, naming_types, group_types, line_ids, positions, label, attention_masks

    def _mixup_augment(self, idx):
        inputs, naming_types, group_types, line_ids, positions, label, attention_masks = self._get_item(idx)

        mixup_idx = torch.randint(0, len(self), (1,)).item()
        (
            mixup_inputs,
            mixup_naming_types,
            mixup_group_types,
            mixup_line_ids,
            mixup_positions,
            mixup_label,
            mixup_attention_masks,
        ) = self._get_item(mixup_idx)

        if len(inputs) > self.max_seq_len // 2:
            inputs = inputs[: self.max_seq_len // 2]
            naming_types = naming_types[: self.max_seq_len // 2]
            positions = positions[: self.max_seq_len // 2]
            attention_masks = attention_masks[: self.max_seq_len // 2]
            line_ids = line_ids[: self.max_seq_len // 2]
            group_types = group_types[: self.max_seq_len // 2]
        if len(mixup_inputs) > self.max_seq_len // 2:
            mixup_inputs = mixup_inputs[: self.max_seq_len // 2]
            mixup_naming_types = mixup_naming_types[: self.max_seq_len // 2]
            mixup_positions = mixup_positions[: self.max_seq_len // 2]
            mixup_attention_masks = mixup_attention_masks[: self.max_seq_len // 2]
            mixup_line_ids = mixup_line_ids[: self.max_seq_len // 2]
            mixup_group_types = mixup_group_types[: self.max_seq_len // 2]

        # last_pos = positions[-1]
        inputs = torch.cat([inputs, mixup_inputs], dim=0)
        naming_types = torch.cat([naming_types, mixup_naming_types], dim=0)
        group_types = torch.cat([group_types, mixup_group_types], dim=0)
        line_ids = torch.cat([line_ids, mixup_line_ids], dim=0)
        positions = torch.arange(len(inputs))
        attention_masks = torch.cat([attention_masks, mixup_attention_masks], dim=0)
        kf = len(inputs) / (len(inputs) + len(mixup_inputs))
        label = (label * kf) + (mixup_label * (1 - kf))
        return inputs, naming_types, group_types, line_ids, positions, label, attention_masks

    def _mask_augment(self, inputs, naming_types, group_types, line_ids, positions, label, attention_masks):
        max_num = 5
        max_len_mask = 10
        max_mask_amont = 0.5

        if len(inputs) < 20:
            return (inputs, naming_types, group_types, line_ids, positions, label, attention_masks)

        num = torch.randint(1, max_num, (1,)).item()
        rest_allowed = int(max_mask_amont * len(inputs))
        for _ in range(num):
            if rest_allowed <= 5:
                break

            mask_len = torch.randint(1, max_len_mask, (1,)).item()
            mask_len = min(mask_len, rest_allowed)
            mask_start = torch.randint(0, len(inputs) - mask_len, (1,)).item()
            mask_end = mask_start + mask_len
            inputs[mask_start:mask_end] = vocab.unk_idx
            attention_masks[mask_start:mask_end] = 0
            naming_types[mask_start:mask_end] = 0
            group_types[mask_start:mask_end] = 0
            line_ids[mask_start:mask_end] = 0

            rest_allowed -= mask_len

        return (inputs, naming_types, group_types, line_ids, positions, label, attention_masks)

    def __getitem__(self, idx):
        if self.aug:
            if torch.rand(1) < 0.2:
                d = self._mixup_augment(idx)
            else:
                d = self._get_item(idx)

            if torch.rand(1) < 0.3:
                d = self._mask_augment(*d)

            return d

        return self._get_item(idx)
