import json
from collections import defaultdict

import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import pairwise_distances
import os
import torch
import datasets
from datasets import Dataset


def cal_nearest_drf(drfs_embs, word_embs) -> dict:
    """
    calculate the distance between drfs_embs and word_embs, return a dictionary of the min distance of each drf
    :param drfs_embs: dict embeddings of drf
    :param word_embs: dict embeddings of word
    :return: min_drf
    """
    min_drf = {k: 1 for k in drfs_embs.keys()}  # get the minimum distance between drfs and NER words
    for word_emb in word_embs:
        for (drf, drf_emb) in drfs_embs.items():
            dis = pairwise_distances(word_emb, drf_emb, metric="euclidean")
            if min_drf[drf] > dis:
                min_drf[drf] = dis
    return min_drf


def run1(num_drf, group_data_root_path, drf_path, annotation_save_path, class_list, ckpt):
    """
    select drf for each sentence to annotate prompt
    :param num_drf: num of drf for each sentence
    :param group_data_root_path: ( pd.DataFrame(columns=['tokens', 'tags', 'class', 'sentence']) )
    :param drf_path: extracted drt save path
    :param annotation_save_path: save path of this function
    :param class_list: MISC...
    :param ckpt: ckpt
    :return: None
    """
    if "roberta" in ckpt.lower():
        tokenizer = AutoTokenizer.from_pretrained(ckpt, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(ckpt, return_dict=True)
    model.eval()
    with open(drf_path) as f:
        drf = json.load(f)
    all_data = datasets.load_from_disk(os.path.join(group_data_root_path, "ALL")).to_pandas()
    print(all_data.info())
    all_drf_embs = defaultdict(dict)
    tag_name = defaultdict(list)
    for domain in class_list:
        tag_name[domain] = ['B-' + domain, 'I-' + domain]
        domain_drf = drf[domain]
        drfs_embs = {k: 0 for k in domain_drf}
        with torch.no_grad():
            for drf_word in domain_drf:
                drf_ids = tokenizer(drf_word, return_tensors='pt')['input_ids'][:, :-1]
                drf_emb = model.get_input_embeddings()(drf_ids).mean(dim=1)
                drfs_embs[drf_word] = drf_emb
        all_drf_embs[domain] = drfs_embs
    annotate_drfs = []
    for i in range(0, all_data.shape[0]):
        word_embs = {domain: [] for domain in class_list}
        # traverse the tags, if the tag belong to tag_name, calculate the distance between the related word and drf
        for index, tag in enumerate(all_data.loc[i]['tags']):
            for domain in class_list:
                if tag in tag_name[domain]:
                    word = all_data.loc[i]['tokens'][index]
                    with torch.no_grad():
                        word_emb = model.get_input_embeddings()(
                            tokenizer(word, return_tensors='pt')['input_ids'][:, :-1]).mean(dim=1)
                    word_embs[domain].append(word_emb)
        min_drf = {}
        # merge the all related drf embedding's distance
        for domain in class_list:
            if len(word_embs[domain]) != 0:
                # print("index:", i, "domain", domain)
                min_drf.update(cal_nearest_drf(all_drf_embs[domain], word_embs[domain]))
        sort_list = sorted(min_drf.items(), key=lambda x: x[1], reverse=False)
        nearest_drf = []
        for index, drf_name in enumerate(sort_list):
            if index == num_drf:
                break
            nearest_drf.append(drf_name[0])
        annotate_drfs.append(nearest_drf)
    all_data['annotate_drf'] = annotate_drfs
    print(f"annotation_dataset{all_data}")
    print(all_data.info())
    Dataset.from_pandas(all_data).save_to_disk(annotation_save_path)


def run(num_drf, group_data_root_path, drf_path, annotation_save_path, class_list, ckpt):
    """
    select drf for each sentence to annotate prompt
    :param num_drf: num of drf for each sentence
    :param group_data_root_path: ( pd.DataFrame(columns=['tokens', 'tags', 'class', 'sentence']) )
    :param drf_path: extracted drt save path
    :param annotation_save_path: save path of this function
    :param class_list: MISC...
    :param ckpt: ckpt
    :return: None
    """
    if "roberta" in ckpt.lower():
        tokenizer = AutoTokenizer.from_pretrained(ckpt, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForTokenClassification.from_pretrained(ckpt, return_dict=True)
    model.eval()
    with open(drf_path) as f:
        drf = json.load(f)
    print(drf)
    print(drf['MISC'])
    annotation_dataset = pd.DataFrame(columns=['tokens', 'tags', 'class', 'sentence', 'annotate_drf'])
    for domain in class_list:
        tag_name = ['B-' + domain, 'I-' + domain]
        print(tag_name)
        group_data_path = os.path.join(group_data_root_path, domain)
        domain_drf = drf[domain]
        count = {k: 0 for k in domain_drf}
        print(f"Domain:{domain},domain_drf:{domain_drf}")
        dataset = datasets.load_from_disk(group_data_path)
        domain_data = dataset.to_pandas()
        print(domain_data.info())
        drfs_embs = {k: 0 for k in domain_drf}
        with torch.no_grad():
            for drf_word in domain_drf:
                drf_ids = tokenizer(drf_word, return_tensors='pt')['input_ids'][:, :-1]
                drf_emb = model.get_input_embeddings()(drf_ids).mean(dim=1)
                drfs_embs[drf_word] = drf_emb
        # print(f"drfs_embs{drfs_embs}")
        annotate_drfs = []
        for i in range(0, domain_data.shape[0]):
            word_embs = []
            # append the NER word's embedding of the domain to word_embs
            for index, tag in enumerate(domain_data.loc[i]['tags']):
                if tag in tag_name:
                    word = domain_data.loc[i]['tokens'][index]
                    with torch.no_grad():
                        word_emb = model.get_input_embeddings()(
                            tokenizer(word, return_tensors='pt')['input_ids'][:, :-1]).mean(dim=1)
                        word_embs.append(word_emb)
            min_drf = cal_nearest_drf(drfs_embs, word_embs)
            sort_list = sorted(min_drf.items(), key=lambda x: x[1], reverse=False)
            # print(sort_list)
            nearest_drf = []
            for index, drf_name in enumerate(sort_list):
                if index == num_drf:
                    break
                nearest_drf.append(drf_name[0])
            for w in nearest_drf:
                count[w] += 1
            annotate_drfs.append(nearest_drf)
            if i < 5:
                print(f"sentence:{domain_data.loc[i]['sentence']}")
                print(f"closest_drfs:", nearest_drf)
        print(f"{domain}\n", count)
        domain_data['annotate_drf'] = annotate_drfs
        # print(annotation_dataset.info())
        # print(domain_data.info())
        annotation_dataset = pd.concat([annotation_dataset, domain_data], ignore_index=True)
        # save_path = os.path.join(annotation_save_path, domain)
        # print("annotated drf dataset", domain_data)
        # Dataset.from_pandas(domain_data).save_to_disk(save_path)
    print(f"annotation_dataset{annotation_dataset}")
    print(annotation_dataset.info())
    Dataset.from_pandas(annotation_dataset).save_to_disk(annotation_save_path)


if __name__ == '__main__':
    run1(40, "../runs/group_data", "../runs/drf.json", "../runs/drf_annotate_data",
         ['PER', 'MISC', 'LOC', 'ORG'], "bert-base-cased")
