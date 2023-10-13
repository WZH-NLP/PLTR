import json
import os.path

import pandas as pd
import datasets as ds
from pathlib import Path
from torch.utils.data import TensorDataset
from datasets import Dataset

PER_TAGS = ['B-PER', 'I-PER']
LOC_TAGS = ['B-LOC', 'I-LOC']
ORG_TAGS = ['B-ORG', 'I-ORG']
MISC_TAGS = ['B-MISC', 'I-MISC']
LABEL_LIST = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC']


def read_augmented_data(path) -> pd.DataFrame:
    """
    read the augmented data
    :param path: file path
    :return: pd.DataFrame
    """
    return ds.load_from_disk(path).to_pandas()


def group_data_by_tag(ori_data: pd.DataFrame, group_data_save_path) -> pd.DataFrame:
    """
    group the data by their tags
    :param ori_data:
    :return: grouped_dataset( pd.DataFrame(columns=['tokens', 'tags', 'class', 'sentence']) )
    """
    new_data = pd.DataFrame(columns=['tokens', 'tags', 'class', 'sentence'])
    all_data = pd.DataFrame(columns=['tokens', 'tags', 'class'])
    count = 0
    for i in range(0, ori_data.shape[0]):
        tokens = ori_data.loc[i]['tokens']
        tags = ori_data.loc[i]['tags']
        class_list = []
        flag = False
        for tag in PER_TAGS:
            if tag in tags:
                flag = True
                class_list.append('PER')
                new_data.loc[len(new_data)] = [tokens, tags, 'PER', ' '.join(tokens)]
                break
        for tag in LOC_TAGS:
            if tag in tags:
                flag = True
                class_list.append('LOC')
                new_data.loc[len(new_data)] = [tokens, tags, 'LOC', ' '.join(tokens)]
                break
        for tag in ORG_TAGS:
            if tag in tags:
                flag = True
                class_list.append('ORG')
                new_data.loc[len(new_data)] = [tokens, tags, 'ORG', ' '.join(tokens)]
                break
        for tag in MISC_TAGS:
            if tag in tags:
                flag = True
                class_list.append('MISC')
                new_data.loc[len(new_data)] = [tokens, tags, 'MISC', ' '.join(tokens)]
                break
        if flag:
            all_data.loc[len(all_data)] = [tokens, tags, class_list]
            count += 1
        # else:
        #     print(tokens)
        #     print(tags)
    print("all_data\n", all_data)
    for s in all_data:
        print(s)
    Dataset.from_pandas(new_data[new_data['class'] == 'PER'].reset_index(drop=True)).save_to_disk(
        os.path.join(group_data_save_path, 'PER'))
    Dataset.from_pandas(new_data[new_data['class'] == 'LOC'].reset_index(drop=True)).save_to_disk(
        os.path.join(group_data_save_path, 'LOC'))
    Dataset.from_pandas(new_data[new_data['class'] == 'ORG'].reset_index(drop=True)).save_to_disk(
        os.path.join(group_data_save_path, 'ORG'))
    Dataset.from_pandas(new_data[new_data['class'] == 'MISC'].reset_index(drop=True)).save_to_disk(
        os.path.join(group_data_save_path, 'MISC'))
    Dataset.from_pandas(all_data).save_to_disk(os.path.join(group_data_save_path, 'ALL'))
    print("\nall_data", all_data.info())
    print("\ncount", count)
    print(f"new_data:\n{new_data}")
    # print(new_data.info())
    # print("------------")
    # print(new_data[new_data['class'] == 'PER'])
    # print(new_data[new_data['class'] == 'PER'].info())
    # x = new_data[new_data['class'] == 'PER']
    # x.reset_index(drop=True, inplace=True)
    # print(x)
    # print(Dataset.from_pandas(x.reset_index(drop=True)))
    # print("------------")
    return new_data


def read_data_from_file(data_path):
    """
    convert the format from a file
    :param data_path:str
    :return: pd.DataFrame(columns=['tokens', 'tags'])
    """
    dataset = {'tokens': [], 'tags': []}
    # print(data_path, os.path.exists(data_path))
    path = Path(data_path)
    raw_texts = path.read_text().strip()
    raw_docs = raw_texts.split('\n\n')
    for i, sent in enumerate(raw_docs):
        sent = sent.split('\n')
        token_list = []
        tag_list = []
        for line in sent:
            line = line.split(' ')
            if len(line) < 2:
                print(path, sent, line)
                continue
            # print(line)
            token, tag = line[0], line[1]
            token_list.append(token)
            tag_list.append(tag)

        dataset['tokens'].append(token_list)
        dataset['tags'].append(tag_list)
    return pd.DataFrame(dataset)


def make_prompt(num) -> list:
    """
    read the drf.json and make the prompt
    :param num: num of each word in prompt
    :return: prompt
    """
    with open("../../runs/drf.json") as f:
        data = json.load(f)
    prompt = ['[DRF]']
    for values in data.values():
        # print(values)
        count = 0
        for word in values:
            if count == num or count > num:
                break
            count += 1
            prompt.append(word)
    return prompt


if __name__ == '__main__':
    loaded_data = read_augmented_data("../../runs/aug_all_data")
    # print(loaded_data)
    # print(loaded_data)
    # print(make_prompt(5))
    group_data_by_tag(loaded_data, "../../runs/group_data")
    # print(read_data_from_file("../../NER_datasets/conll2003/train.txt"))
    # a = make_prompt(0)
    # print(a, len(a))
