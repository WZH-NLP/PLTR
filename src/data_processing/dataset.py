import copy
import os
from collections import defaultdict

import torch
from torch import LongTensor, Tensor
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

from src.data_processing import process_data


class NerDataset(Dataset):

    def __init__(self,
                 data_type: str,  # train val test
                 tokenizer,  # tokenizer
                 data_path,
                 # 1:train: (drf annotated data {columns=['tokens', 'tags', 'class', 'sentence', 'annotate_drf']})
                 # 2:val: NER_datasets/conll2003/dev.txt
                 # 3:test: NER_datasets
                 label_list,  # ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
                 mask_num,  # number of [MASK]
                 max_len: int = 256,  # tokenizer max length
                 test_domain=None
                 ):
        self.data_type = data_type
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_path = data_path
        self.label_list = label_list
        self.test_domain = test_domain
        self.label2id = {label: index for index, label in enumerate(label_list)}
        self.id2label = {index: label for label, index in self.label2id.items()}
        self.mask_num = mask_num
        self.data = self._init_data()

    def align_label(self, tokens: list, labels: list):
        # print(tokens)
        # print(labels)
        tokenized_inputs = self.tokenizer(tokens, max_length=self.max_len,
                                          truncation=True,
                                          is_split_into_words=True,
                                          padding='max_length')
        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(self.label2id[labels[word_idx]])
                except:
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(self.label2id[labels[word_idx]])
                except:
                    label_ids.append(-100)
            previous_word_idx = word_idx
        # print('1', label_ids)
        return label_ids

    def _init_data(self):
        data = defaultdict(list)
        if self.data_type == 'train':
            ori_data = load_from_disk(self.data_path)
            ori_data = ori_data.to_pandas()
        else:
            if self.data_type == 'val':
                path = self.data_path
            else:
                path = os.path.join(self.data_path, self.test_domain, "test.txt")
            ori_data = process_data.read_data_from_file(path)
        for index in range(0, ori_data.shape[0]):
            if self.data_type == 'train':
                tokens = ori_data.loc[index]['tokens'].tolist()
            else:
                tokens = ori_data.loc[index]['tokens']
            # print(tokens)
            # print(type(tokens))
            data['labels'].append(LongTensor(self.align_label(tokens, ori_data.loc[index]['tags'])))
            # print("**", data['labels'][index])
            tokens.extend([self.tokenizer.sep_token])
            ori_tokens = copy.deepcopy(tokens)
            for i in range(self.mask_num):
                tokens.append(self.tokenizer.mask_token)
                # tokens.append(',')
            tokenized_data = self.tokenizer(tokens,
                                            max_length=self.max_len,
                                            truncation=True,
                                            # return_tensors="pt",
                                            is_split_into_words=True,
                                            padding='max_length', )
            data['input_ids'].append(LongTensor(tokenized_data['input_ids']))
            # print('2', tokenized_data['input_ids'])
            data['attention_mask'].append(LongTensor(tokenized_data['attention_mask']))
            # print('3', tokenized_data['attention_mask'])
            if self.data_type == 'train':
                drf_label = []
                drf_labels = []
                drf_list = ori_data.loc[index]['annotate_drf'].tolist()
                # print(drf_list[0:self.mask_num])
                # print(len(drf_list[0:self.mask_num]))
                golden_input = ori_tokens + drf_list[0:self.mask_num]
                data['golden_input_ids'].append(LongTensor(self.tokenizer(golden_input, max_length=self.max_len,
                                                                          truncation=True,
                                                                          # return_tensors="pt",
                                                                          is_split_into_words=True,
                                                                          padding='max_length')['input_ids']))
                for drf in drf_list:
                    tokenized_drf = self.tokenizer(drf,
                                                   # max_length=self.max_len,
                                                   truncation=True,
                                                   # return_tensors="pt",
                                                   # padding='max_length'
                                                   )
                    # append the first token
                    drf_label.append(tokenized_drf['input_ids'][1])
                # print(drf_label)
                # print(tokenized_data['input_ids'])
                for item in tokenized_data['input_ids']:
                    if item == self.tokenizer.mask_token_id:
                        if len(drf_label) != 0:
                            token = drf_label.pop(0)
                        else:
                            token = -100
                        drf_labels.append(token)
                    else:
                        drf_labels.append(-100)
                data['drf_labels'].append(LongTensor(drf_labels))
                # print('4', drf_labels
        return data

    def __getitem__(self, item):
        return {
            k: v[item]
            for k, v in self.data.items()
        }

    def __len__(self):
        return len(self.data['input_ids'])


def test_dataset():
    label_list = ['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    train_dataset = NerDataset('train', tokenizer, "../../runs/drf_annotate_data", label_list, 10, 256)
    val_dataset = NerDataset('val', tokenizer, "../../NER_datasets/conll2003/dev.txt", label_list, 10, 256)
    test_dataset = NerDataset('test', tokenizer, "../../NER_datasets", label_list, 10, 256, "ai")
    for batch in DataLoader(train_dataset, batch_size=2, shuffle=True):
        print("\n1", batch['input_ids'])  # tensor([ [],[] ])
        print("\n2", batch['labels'])  # tensor([ [],[] ])
        print("\n3", batch['attention_mask'])  # tensor([ [],[] ])
        print("\n4", batch['drf_labels'])  # tensor([ [],[] ])
        print("\n5", batch['ori_input_ids'])  # tensor([ [],[] ])
        print("\n6", batch['golden_input_ids'])  # tensor([ [],[] ])
        # print("\n6", torch.masked_select(batch['drf_labels'], batch['input_ids'] == tokenizer.mask_token_id))


if __name__ == '__main__':
    test_dataset()
