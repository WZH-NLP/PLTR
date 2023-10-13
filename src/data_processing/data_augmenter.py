import torch
import numpy as np
from datasets import load_from_disk, load_metric, Dataset
from transformers import pipeline, AutoModelForTokenClassification, Trainer, TrainingArguments, \
    DataCollatorForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader
import os
import pandas as pd
import datasets
import copy
import random
from src.data_processing import process_data
from sklearn import metrics

global tokenizer
global tag2id


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    examples["tag_ids"] = [[tag2id[tag] for tag in tags] for tags in examples["tags"]]
    for i, label in enumerate(examples["tag_ids"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


class DataAugmenter:
    def __init__(self,
                 ckpt,  # fill mask model and ori token classifier model
                 batch_size,
                 output_pad_token,
                 num_samples,  # numbers of ori train set
                 label_list,  # all ner labels
                 all_train_dataset=None,
                 sampled_trainset_save_path="../../runs/ori_train_dataset",
                 ori_model_save_path="../../runs/ori_model",
                 semi_ratio=-1,
                 counter_ratio=-1
                 ):

        global tokenizer
        if "roberta" in ckpt.lower():
            tokenizer = AutoTokenizer.from_pretrained(ckpt, add_prefix_space=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
        print(tokenizer.mask_token)
        global tag2id
        tag2id = {tag: id for id, tag in enumerate(label_list)}
        self.ckpt = ckpt
        self.label2id = tag2id
        print(self.label2id)
        self.id2label = {id: tag for tag, id in tag2id.items()}
        self.input_pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.output_pad_id = tag2id[output_pad_token]
        self.output_pad_token = output_pad_token
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.label_list = label_list
        self.num_samples = num_samples
        self.all_train_dataset = all_train_dataset
        self.sampled_trainset_save_path = sampled_trainset_save_path
        self.batch_size = batch_size
        self.ori_model_save_path = ori_model_save_path
        self.semi_ratio = semi_ratio
        self.counter_ratio = counter_ratio
        print("\n\n----------", "init augmenter model and sampled trainset", "----------")
        self.sampled_trainset, self.model = self.__init_model_and_sampled_trainset()

    def __init_model_and_sampled_trainset(self):
        """
        init the model(used when check the augmented data) and sample the train set (num_samples shot per class)
        :return: a model and a sampled trainset
        """

        def compute_metrics(p):
            metric = load_metric("src/utils/my_seqeval.py")
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        def train(save_path, trainset, devset, data_collator):
            if os.path.exists(save_path):
                print('\n\nLoading from', save_path, "\n\n")
                model = AutoModelForTokenClassification.from_pretrained(save_path)
                load = True
            else:
                os.makedirs(save_path)
                print('\n\nTraining and saving to', save_path, "\n\n")
                model = AutoModelForTokenClassification.from_pretrained(self.ckpt, num_labels=len(self.label_list))
                load = False

            args = TrainingArguments(
                'model_cache',
                evaluation_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=2 * self.batch_size,
                num_train_epochs=3,
                weight_decay=0,
                push_to_hub=False,
            )

            trainer = Trainer(
                model,
                args,
                train_dataset=trainset,
                eval_dataset=devset,
                data_collator=data_collator,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics
            )
            if not load:
                trainer.train()
                trainer.evaluate()
                trainer.model.save_pretrained(save_path)
            return trainer

        if not os.path.exists(self.sampled_trainset_save_path):
            if self.all_train_dataset is None:
                train_set = process_data.read_data_from_file("NER_datasets/conll2003/train.txt")
                print("\n\nnot sampled train set:\n", train_set)
            else:
                train_set = datasets.load_from_disk(self.all_train_dataset)
                train_set = train_set.to_pandas()
            sampled_trainset = self.sampling(Dataset.from_pandas(train_set))
            print("\n\nsampled train set:\n", sampled_trainset)
            sampled_trainset = pd.DataFrame(sampled_trainset)
            sampled_trainset = datasets.Dataset.from_pandas(sampled_trainset)
            dataset = pd.DataFrame(columns=['tokens', 'tags'])
            for i, example in enumerate(sampled_trainset):
                dataset.loc[i, :] = [example['tokens'], example['tags']]
            ori_train_data = datasets.Dataset.from_pandas(dataset)
            ori_train_data = ori_train_data.map(tokenize_and_align_labels, batched=True)
            print("\n\nsave sampled train set to", self.sampled_trainset_save_path, "\n\n")
            ori_train_data.save_to_disk(self.sampled_trainset_save_path)
        else:
            print("\n\nload sampled train set from", self.sampled_trainset_save_path, "\n\n")
            ori_train_data = load_from_disk(self.sampled_trainset_save_path)
        dev_set = datasets.Dataset.from_pandas(process_data.read_data_from_file("NER_datasets/conll2003/dev.txt"))
        dev_set = dev_set.map(tokenize_and_align_labels, batched=True)
        print("\n\n----------", "start train augment model", "----------")
        ori_trainer = train(self.ori_model_save_path, ori_train_data, dev_set,
                            data_collator=DataCollatorForTokenClassification(tokenizer)
                            )
        return ori_train_data, ori_trainer.model

    def sampling(self, data):  # num_samples per tag
        seed = random.randint(1, 1000)
        data = data.shuffle(seed=seed)
        if self.num_samples == -1:
            return data
        count = np.zeros((len(self.label_list),), dtype=np.int64)
        sampled_data = []

        for d in data:
            idx = np.where(count < self.num_samples)[0]
            id_list = [self.label2id[tag] for tag in d['tags']]
            if len(set(idx).intersection(set(id_list))) == 0:  # There is no currently required tag
                continue
            sampled_data.append(d)

            for label in d['tags']:
                count[self.label2id[label]] += 1

        return sampled_data

    def collate_fn(self, input_pad_id, output_pad_id, device):

        def collate_fn_wrapper(batch):
            max_seq_len = 36
            # truncation
            for i, _ in enumerate(batch):
                if len(batch[i]["input_ids"]) > max_seq_len:
                    batch[i]["input_ids"] = batch[i]["input_ids"][:max_seq_len]
                    batch[i]["labels"] = batch[i]["labels"][:max_seq_len]
                    batch[i]["attention_mask"] = batch[i]["attention_mask"][:max_seq_len]
            # padding
            for i, _ in enumerate(batch):
                length = len(batch[i]["input_ids"])
                batch[i]["input_ids"] += [input_pad_id] * (max_seq_len - length)
                batch[i]["labels"] += [output_pad_id] * (max_seq_len - length)
                batch[i]["attention_mask"] = [1] * length + [0] * (max_seq_len - length)

            input_ids = torch.LongTensor([sample['input_ids'] for sample in batch]).to(device)
            labels = torch.LongTensor([sample['labels'] for sample in batch]).to(device)
            masks = torch.LongTensor([sample['attention_mask'] for sample in batch]).to(device)
            return input_ids, labels, masks

        return collate_fn_wrapper

    def comp(self, ori_model, sampled_trainset, reasonable_aug_examples,
             input_pad_id, output_pad_id):

        ori_loader = DataLoader(sampled_trainset, batch_size=self.batch_size, shuffle=False,
                                collate_fn=self.collate_fn(input_pad_id, output_pad_id, self.device))
        gen_loader = DataLoader(reasonable_aug_examples, batch_size=self.batch_size, shuffle=False,
                                collate_fn=self.collate_fn(input_pad_id, output_pad_id, self.device))
        ori_hidden = []
        gen_hidden = []
        for batch in ori_loader:
            input_ids, batch_labels, masks = batch
            outputs = ori_model(input_ids=input_ids, attention_mask=masks, output_hidden_states=True)
            ori_hidden.extend(outputs.hidden_states[0].detach().cpu().tolist())

        for batch in gen_loader:
            input_ids, batch_labels, masks = batch
            outputs = ori_model(input_ids=input_ids, attention_mask=masks, output_hidden_states=True)
            gen_hidden.extend(outputs.hidden_states[0].detach().cpu().tolist())

        ori_gen_pair = []
        last_ori_sent = None
        for i, example in enumerate(reasonable_aug_examples):
            ori_index = example['ori_index']
            ori_h = np.array(ori_hidden[ori_index])
            gen_h = np.array(gen_hidden[i])
            ori_gen_pair.append((ori_h, gen_h))

        gen_rank = []
        for (ori, gen) in ori_gen_pair:
            ori, gen = np.array(ori), np.array(gen)
            NMI = metrics.normalized_mutual_info_score(ori.reshape((-1,)), gen.reshape((-1,)))
            gen_rank.append(NMI)
        # get the sorted index of the element in the original array
        # the index of smaller NMI scores will be presented first
        # i.e., in variable 'order', we should retain last aug_num samples for a larger NMI score
        order = np.argsort(np.array(gen_rank))

        return order

    def create_counterfactual_examples(self, trainset, pad_tag, aug_num=50):
        deduplicated_examples = set()
        counterfactual_examples = []
        local_entity_sets = {}

        for example in trainset:
            deduplicated_examples.add(copy.deepcopy(' '.join(example["tokens"])))
            for token, tags in zip(example["tokens"], example['tags']):
                if tags == pad_tag:
                    continue
                if tags in local_entity_sets.keys():
                    local_entity_sets[tags].append(token)
                else:
                    local_entity_sets[tags] = [token]
        for key in local_entity_sets.keys():
            local_entity_sets[key] = list(set(local_entity_sets[key]))
        local_entity_sets[pad_tag] = []
        for i, example in enumerate(trainset):
            local_entity_subsets = []
            count = 0
            while len(local_entity_subsets) < 1 and count < len(example['tokens']):
                index = random.choice(list(range(len(example['tokens']))))
                count += 1
                if len(local_entity_sets[example["tags"][index]]) > 0:
                    local_entity_subsets = local_entity_sets[example["tags"][index]]

            for j, local_candidate in enumerate(local_entity_subsets):
                cfexample = copy.deepcopy(example)
                cfexample["ori_index"] = i
                cfexample["obersavational_text"] = example["tokens"]
                cfexample["tokens"][index] = local_candidate
                if cfexample["tokens"] == example["tokens"] or ' '.join(cfexample["tokens"]) in deduplicated_examples:
                    continue

                deduplicated_examples.add(copy.deepcopy(' '.join(cfexample["tokens"])))
                cfexample["replaced"] = [
                    "[{0}]({1}, {2})".format(
                        cfexample["tokens"][index], index, example["tags"][index].split('-')[1]
                    )
                ]
                counterfactual_examples.append(cfexample)
                # if j % aug_num == 0:
                #     print(j)
        return counterfactual_examples, deduplicated_examples

    def create_semifactual_examples(self, trainset, pad_tag, aug_num=50):
        deduplicated_examples = set()
        counterfactual_examples = []
        # filler = SubstituteWithBert()
        filler = pipeline('fill-mask', model=self.ckpt, tokenizer=self.ckpt, device=0)
        for i, example in enumerate(trainset):

            for index in range(len(example['tokens'])):
                if example['tags'][index] != pad_tag:  # we only substitute the Non-O token
                    continue

                cfexample = copy.deepcopy(example)
                cfexample["ori_index"] = i
                cfexample["obersavational_text"] = example["tokens"]
                # mask-and-fill
                # cfexample["input_ids"][index+1] = 103 # [MASK] ~ 103, first place of input_ids is [CLS]
                cfexample["tokens"][index] = tokenizer.mask_token
                fill_result = filler(' '.join(cfexample["tokens"]), top_k=1)
                candidate = fill_result[0]['token_str']
                cfexample["tokens"][index] = candidate

                if cfexample["tokens"] == example["tokens"] or ' '.join(cfexample["tokens"]) in deduplicated_examples:
                    continue

                deduplicated_examples.add(copy.deepcopy(' '.join(cfexample["tokens"])))
                cfexample["replaced"] = [
                    "[{0}]({1}, {2})".format(
                        cfexample["tokens"][index], index, example["tags"][index]
                    )
                ]
                counterfactual_examples.append(cfexample)
                # if j % aug_num == 0:
                #     print(j)
        return counterfactual_examples, deduplicated_examples

    def check_data(self, model, all_aug_examples, aug_dataloader, id2tag) -> pd.DataFrame:
        reasonable_aug_examples = []
        # check if reasonable
        for i, batch in enumerate(aug_dataloader):
            input_ids, labels, masks = batch  # 32 sents
            outputs = model(input_ids=input_ids, attention_mask=masks)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, -1)
            preds = probs.argmax(-1).detach().cpu().tolist()
            for j, pred in enumerate(preds):  # 32 sents, pred ~ 1 sent
                tokens = all_aug_examples[i * self.batch_size + j]["tokens"]
                tags = all_aug_examples[i * self.batch_size + j]["tags"]
                replaced_spans = all_aug_examples[i * self.batch_size + j]["replaced"]
                predicted_spans = ["[{0}]({1}, {2})".format(token, index, id2tag[pred_id].split('-')[1] if id2tag[
                                                                                                               pred_id] != 'O' else 'O')
                                   for index, (token, pred_id) in enumerate(zip(tokens, pred))]
                if len(set(replaced_spans).intersection(set(predicted_spans))) == len(replaced_spans):
                    reasonable_aug_examples.append(all_aug_examples[i * self.batch_size + j])
                    # print('add')
        reasonable_aug_examples = pd.DataFrame(reasonable_aug_examples)
        # reasonable_aug_examples = datasets.Dataset.from_pandas(reasonable_aug_examples)

        return reasonable_aug_examples

    def augment(self,
                # aug_ratio,
                save_path,
                is_semi=False):
        # if aug_ratio == 0:
        #     return []
        if is_semi:
            print("creating semifactual examples...")
            all_aug_examples, deduplicated_examples = self.create_semifactual_examples(self.sampled_trainset,
                                                                                       self.output_pad_token)
        else:
            print("creating counterfactual examples...")
            all_aug_examples, deduplicated_examples = self.create_counterfactual_examples(self.sampled_trainset,
                                                                                          self.output_pad_token)
        all_aug_examples = pd.DataFrame(all_aug_examples)
        all_aug_examples = datasets.Dataset.from_pandas(all_aug_examples)
        all_aug_examples = all_aug_examples.map(tokenize_and_align_labels, batched=True)
        # to check if the generated cf examples are linguistically reasonable
        aug_dataloader = DataLoader(all_aug_examples, self.batch_size, shuffle=True,
                                    collate_fn=self.collate_fn(self.input_pad_id, self.output_pad_id,
                                                               self.device))
        reasonable_aug_examples = self.check_data(self.model, all_aug_examples,
                                                  aug_dataloader,
                                                  self.id2label)
        print(f'{len(self.sampled_trainset)=}')
        print(f'{len(all_aug_examples)=}')
        print(f'{len(reasonable_aug_examples)=}')
        maximum_ratio = len(reasonable_aug_examples) / len(self.sampled_trainset)
        print('maximum ratio:', maximum_ratio)
        if is_semi:
            if self.semi_ratio != -1:
                ratio = self.semi_ratio if self.semi_ratio < maximum_ratio else maximum_ratio
                num = int(ratio * len(self.sampled_trainset))
                reasonable_aug_examples = reasonable_aug_examples.sample(n=num).reset_index(drop=True)
        else:
            if self.counter_ratio != -1:
                ratio = self.counter_ratio if self.counter_ratio < maximum_ratio else maximum_ratio
                num = int(ratio * len(self.sampled_trainset))
                reasonable_aug_examples = reasonable_aug_examples.sample(n=num).reset_index(drop=True)
        Dataset.from_pandas(reasonable_aug_examples).save_to_disk(save_path)
        return reasonable_aug_examples
        # return reasonable_aug_examples
        # maximum = int(len(reasonable_aug_examples) / len(self.sampled_trainset))
        # # sampling the augmented examples using MMI
        # sorting_index = self.comp(self.model, self.sampled_trainset, reasonable_aug_examples, self.input_pad_id,
        #                           self.output_pad_id)
        #
        # for aug_ratio in range(1, maximum + 1):
        #     aug_num = aug_ratio * len(self.sampled_trainset)
        #     aug_num = int(aug_num) if aug_num < len(reasonable_aug_examples) else len(reasonable_aug_examples)
        #     print('ori samples:', len(self.sampled_trainset), 'aug samples:', aug_num)
        #     path = aug_example_path.replace('aug_ratio', str(aug_ratio))
        #     sampling_index = sorting_index[-aug_num:]
        #     selected_aug_examples = reasonable_aug_examples.select(sampling_index)
        #     remove_columns = set(selected_aug_examples.features) ^ set(self.sampled_trainset.features)
        #     selected_aug_examples = selected_aug_examples.remove_columns(remove_columns)
        #     # selected_aug_examples.save_to_disk(path)
        #
        # aug_num = aug_ratio * len(self.sampled_trainset)
        # sampling_index = sorting_index[-aug_num:]
        # selected_aug_examples = reasonable_aug_examples.select(sampling_index)
        #
        # return selected_aug_examples

    def run(self, semi_data_save_path, counter_data_save_path, all_data_save_path):
        semi_data = self.augment(semi_data_save_path, True)
        print("\n\n----------semi data----------\n", semi_data)
        counter_data = self.augment(counter_data_save_path, False)
        print("\n\n----------counter data----------\n", counter_data)
        merge_data = pd.concat([semi_data, counter_data], ignore_index=True)
        sampled_examples = merge_data[['tokens', 'tags']]
        ori_examples = load_from_disk(self.sampled_trainset_save_path)
        print("\n\n----------sampled ori examples----------\n", ori_examples)
        all_examples = pd.concat([ori_examples.to_pandas(), sampled_examples], ignore_index=True)
        print("\n\n----------all examples----------\n", all_examples)
        Dataset.from_pandas(all_examples).save_to_disk(all_data_save_path)
