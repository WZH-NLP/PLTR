import json
import os.path
from collections import defaultdict

import numpy as np
from pytorch_lightning import LightningModule
from torch import LongTensor
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig, BatchEncoding, AdamW, AutoModel, \
    get_linear_schedule_with_warmup, get_scheduler
import torch.nn
from src.data_processing.dataset import NerDataset
from datasets import Metric, load_metric
from torch.utils.data import Dataset, DataLoader
import random


class PLTRModel(LightningModule):
    def __init__(self,
                 ckpt,
                 hidden_dropout_prob,
                 # dataset setting
                 train_data_path,
                 val_data_path,
                 label_list,
                 mask_num,
                 max_len,
                 drf_constant,
                 learning_rate,
                 train_batch_size,
                 val_batch_size,
                 metric,
                 drf_path,
                 max_epoch,
                 scheduler_type
                 ):
        super().__init__()
        self.name = "CDFWNerModel1Add, add per,loc.. before prompt"
        self.ckpt = ckpt
        self.max_len = max_len
        self.encoder = AutoModel.from_pretrained(self.ckpt)
        self.scheduler_type=scheduler_type
        if "roberta" in self.ckpt.lower():
            self.mlm_head = AutoModelForMaskedLM.from_pretrained(self.ckpt).lm_head
            self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt, add_prefix_space=True)
        else:
            self.mlm_head = AutoModelForMaskedLM.from_pretrained(self.ckpt).cls
            self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, len(label_list))
        self.label_list = label_list
        self.mask_num = mask_num
        self.drf_constant = drf_constant
        self.label2id = {label: index for index, label in enumerate(label_list)}
        self.id2label = {index: label for label, index in self.label2id.items()}
        self.train_datasets, self.val_datasets = self._init_datasets(train_data_path, val_data_path, max_len)
        self.learning_rate = learning_rate
        # self.loss_fct = CrossEntropyLoss()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.metric = metric
        self.all_results = []
        self.max_score = defaultdict(int)
        self.drf_first_id2word, self.drf_first_id2ids, self.drf_word2ids, self.drf_word2domain = self._init_drf_info(
            drf_path)
        self.current_predict_word_ids = []
        self.max_epoch = max_epoch

    def _init_drf_info(self, drf_path):
        with open(drf_path) as f:
            drf = json.load(f)
        drf_word2domain = {}
        for item in drf.items():
            for word in item[1]:
                drf_word2domain[word] = item[0]
        drf_list = []
        for v in drf.values():
            drf_list.extend(v)
        drf_first_id_list = [self.tokenizer(word, truncation=True)['input_ids'][1] for word in drf_list]
        drf_first_id2word = {drf_first_id: word for drf_first_id, word in zip(drf_first_id_list, drf_list)}
        drf_first_id2ids = {
            drf_first_id: self.tokenizer(drf_first_id2word[drf_first_id], truncation=True)['input_ids'][1:-1] for
            drf_first_id in drf_first_id_list}
        drf_word2ids = {drf_first_id2word[word_id]: drf_first_id2ids[word_id] for word_id in drf_first_id_list}
        return drf_first_id2word, drf_first_id2ids, drf_word2ids, drf_word2domain

    def _init_datasets(self, train_data_path, val_data_path, max_len):
        train = NerDataset("train", self.tokenizer, data_path=train_data_path, label_list=self.label_list,
                           mask_num=self.mask_num, max_len=max_len)
        val = NerDataset("val", self.tokenizer, data_path=val_data_path, label_list=self.label_list,
                         mask_num=self.mask_num, max_len=max_len)
        return train, val

    def _get_max_score_word_ids(self, logits, position):
        """
        find the most relevant drf in logits
        :param logits: Tensor(batch_size, sequence_length, config.vocab_size)
        :param position: list [int, int]
        """
        predict = logits[position[0]][position[1]]
        max_score = -1000
        word_id = 0
        for ids in self.drf_first_id2ids.keys():
            if ids in self.current_predict_word_ids:
                continue
            if predict[ids] > max_score:
                max_score = predict[ids]
                word_id = ids
        self.current_predict_word_ids.append(word_id)
        return self.drf_first_id2word[word_id], self.drf_first_id2ids[word_id]

    def _get_prompt_input_ids(self, word_list):
        """
        add prompt template based on word list
        :param word_list: selected drf words
        :return template word list
        """
        domain_name2domain_label = {"LOC": "location", "PER": "person", "MISC": "miscellaneous", "ORG": "organization"}
        template_list = []
        word_dict = defaultdict(list)  # location:[,,,]...
        for word in word_list:
            word_dict[domain_name2domain_label[self.drf_word2domain[word]]].extend(self.drf_word2ids[word])
        for item in word_dict.items():
            template_list.extend(self.tokenizer(item[0])['input_ids'][1:-1])
            template_list.extend(self.tokenizer("is")['input_ids'][1:-1])
            template_list.extend(item[1])
            template_list.extend(self.tokenizer(self.tokenizer.sep_token)['input_ids'][1:-1])
        # print(template_list)
        return template_list

    def forward(self, input_ids, attention_mask, labels, drf_labels, batch_index, **kwargs):
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        prediction_scores = self.mlm_head(hidden_states)
        masked_lm_loss = 0
        mask_token_index = (input_ids == self.tokenizer.mask_token_id).nonzero().tolist()
        predict_word_ids = defaultdict(list)
        predict_words = defaultdict(list)
        # get all tokens of predict words
        current_sentence_index = -1
        for position in mask_token_index:
            if position[0] != current_sentence_index:
                current_sentence_index = position[0]
                self.current_predict_word_ids = []
            word, word_ids = self._get_max_score_word_ids(prediction_scores, position)
            predict_word_ids[position[0]].extend(word_ids)
            predict_words[position[0]].append(word)
        batch_size = self.val_batch_size
        if drf_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.encoder.config.vocab_size), drf_labels.view(-1))
            batch_size = self.train_batch_size  # label_id = torch.masked_select(drf_labels, input_ids == self.tokenizer.mask_token_id)
            # if batch_index % 5 == 0:
            #     label_id = torch.where(label_id == -100, 101, label_id)
            #     self.print("label_words", self.tokenizer.convert_ids_to_tokens(label_id))
        prompt_input_ids = []
        input_id_list = input_ids.tolist()
        for index in range(0, batch_size):
            sentence = input_id_list[index]
            temp = []
            for word_id in sentence:
                # copy the word before [MASK]
                if word_id == self.tokenizer.mask_token_id:
                    break
                temp.append(word_id)
            temp.extend(self._get_prompt_input_ids(predict_words[index]))
            if len(temp) < self.max_len:
                prompt_input_ids.append(temp + (self.max_len - len(temp)) * [0])
            else:
                prompt_input_ids.append(temp[0:self.max_len])
        # if batch_index % 5 == 0:
        #     self.print(predict_words)
        #     predict_id = torch.where(predict_id == -100, 101, predict_id)
        #     self.print("predict_words", self.tokenizer.convert_ids_to_tokens(predict_id))
        cls_attention_mask = []
        for item in prompt_input_ids:
            temp = []
            for input_id in item:
                if input_id != 0:
                    temp.append(1)
                else:
                    temp.append(0)
            cls_attention_mask.append(temp)
        cls_hidden_states = \
            self.encoder(input_ids=LongTensor(prompt_input_ids).to('cuda'),
                         attention_mask=attention_mask)[0]
        sequence_output = self.dropout(cls_hidden_states)
        cls_logits = self.classifier(sequence_output.to('cuda'))
        loss_fct = CrossEntropyLoss()
        cls_loss = loss_fct(cls_logits.view(-1, len(self.label_list)), labels.view(-1))
        loss = cls_loss + (self.drf_constant * masked_lm_loss)
        # loss = cls_loss
        return {"loss": loss, "logits": cls_logits, "cls_loss": cls_loss, "masked_lm_loss": masked_lm_loss}

    def training_step(self, batch: BatchEncoding, batch_idx):
        output = self(input_ids=batch['input_ids'],
                      attention_mask=batch['attention_mask'],
                      labels=batch['labels'],
                      drf_labels=batch['drf_labels'],
                      batch_index=batch_idx)
        # self.log_dict(output, on_step=False, on_epoch=False, prog_bar=True, logger=True)
        # self.print("batch index:", batch_idx, "\nloss:", output["loss"], "\ncls_loss:", output["cls_loss"],
        #            "\nmasked_lm_loss:", output["masked_lm_loss"], "\n\n\n")

        loss = output['loss']
        logits = output['logits']
        pred_ids = logits.detach().cpu().argmax(dim=-1).tolist()
        label_ids = batch['labels'].cpu().tolist()
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_ids, label_ids)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_ids, label_ids)
        ]

        return dict(
            loss=output['loss'],
            cls_loss=output['cls_loss'],
            masked_lm_loss=output['masked_lm_loss'],
            true_predictions=true_predictions,
            true_labels=true_labels
        )

    def validation_step(self, batch: BatchEncoding, batch_idx, dataloader_idx):
        return self._eval_step(batch, batch_idx, dataloader_idx)

    def test_step(self, batch: BatchEncoding, batch_idx, dataloader_idx):
        return self._eval_step(batch, batch_idx, dataloader_idx)

    # def _get_prompt_input_ids(self,input_ids:LongTensor,predicted_token_ids):

    def _eval_step(self, batch, batch_idx, dataloader_idx):

        # mlm_output = self.mask_lm(batch['input_ids'], batch['attention_mask'])
        # logits = mlm_output.logits.argmax(dim=-1)
        # prompt_input_ids = torch.where(batch['input_ids'] == self.tokenizer.mask_token_id, logits, batch['input_ids'])

        prompt_input_ids = batch['input_ids']
        outputs = self(input_ids=batch['input_ids'],
                       attention_mask=batch['attention_mask'],
                       labels=batch['labels'],
                       drf_labels=None,
                       batch_index=batch_idx)
        loss = outputs['loss']
        logits = outputs['logits']
        pred_ids = logits.detach().cpu().argmax(dim=-1).tolist()
        label_ids = batch['labels'].cpu().tolist()
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_ids, label_ids)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_ids, label_ids)
        ]
        return dict(
            true_predictions=true_predictions,
            true_labels=true_labels,
            loss=loss,
            input_ids=batch['input_ids'],
            prompt_input_ids=prompt_input_ids
        )

    def training_epoch_end(self, outputs):
        avg_epoch_loss = torch.stack([batch["loss"] for batch in outputs]).mean()
        avg_epoch_cls_loss = torch.stack([batch["cls_loss"] for batch in outputs]).mean()
        avg_epoch_masked_lm_loss = torch.stack([batch["masked_lm_loss"] for batch in outputs]).mean()
        # self.log(f"avg_train_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.print("\n\n\n<<<----------training_epoch_end----------->>>")
        self.print("avg_train_loss", avg_epoch_loss, "\n")
        self.print("avg_train_cls_loss", avg_epoch_cls_loss, "\n")
        self.print("avg_train_masked_lm_loss", avg_epoch_masked_lm_loss, "\n")

        train_eval_dict = defaultdict(list)
        for batch_eval_dict in outputs:
            for k, v in batch_eval_dict.items():
                if isinstance(v, list):
                    train_eval_dict[k].extend(v)
                else:
                    train_eval_dict[k].append(v)
        self.print("\n<<<----------Results during train----------->>>")
        results = self._eval_metrix(train_eval_dict['true_predictions'], train_eval_dict['true_labels'])

    def _eval_epoch_end(self, outputs):
        # self.print("eval epoch end\n\n\n")
        # avg_epoch_loss = torch.stack([batch["loss"] for batch in outputs]).mean()
        # # self.log(f"avg_train_loss", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.print("\navg_eval_loss", avg_epoch_loss)
        results = []
        for idx, output in enumerate(outputs):  # all dataloaders
            print("\n\n\n\n<<<----------eval index----------->>>", idx)
            epoch_eval_dict = defaultdict(list)
            for batch_eval_dict in output:
                for k, v in batch_eval_dict.items():
                    if isinstance(v, list):
                        epoch_eval_dict[k].extend(v)
                    else:
                        epoch_eval_dict[k].append(v)
            results.append(self._eval_metrix(epoch_eval_dict['true_predictions'], epoch_eval_dict['true_labels']))
        self.all_results.append(results)
        print("\n<<<---------------------------------->>>")
        print("<<<---------------------------------->>>")
        print("<<<---------------------------------->>>\n\n")
        total = 0
        for idx, result in enumerate(results):
            print(idx, "--->", result['f1'])
            total += result['f1']
            if result['f1'] > self.max_score[idx]:
                self.max_score[idx] = result['f1']
        print("total_score--->", total)

    def write_to_file(self, path):
        with open(path, 'w') as f:
            print(self.max_score, file=f)
            for idx, result in enumerate(self.all_results):
                print("\n\nepoch---->:", idx, file=f)
                for i, domain_result in enumerate(result):
                    print("    domain:", i, file=f)
                    f.write("    " + json.dumps(domain_result) + '\n')

    # def get_current_f1_score(self):
    #     return self.results['f1']

    def _eval_metrix(self, predicts, labels):
        results = self.metric.compute(predictions=predicts, references=labels)
        self.print("<<<----------print results----------->>>")
        self.print(results)
        result_dict = {"precision": results["overall_precision"],
                       "recall": results["overall_recall"],
                       "f1": results["overall_f1"],
                       "accuracy": results["overall_accuracy"]}
        return result_dict

    def test_epoch_end(self, outputs):
        self.print("\n\n\n<<<----------test_epoch_end----------->>>")
        self._eval_epoch_end(outputs)

    def validation_epoch_end(self, outputs):
        self.print("\n\n\n<<<----------validation_epoch_end----------->>>")
        self._eval_epoch_end(outputs)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_grouped_parameters, lr=self.learning_rate, eps=1e-8)
        t_total = (len(self.train_datasets) // self.train_batch_size) * self.max_epoch
        scheduler = get_scheduler(self.scheduler_type,
                                  opt,
                                  num_warmup_steps=0,
                                  num_training_steps=t_total)
        print("scheduler", scheduler)
        print("opt", opt)
        return {"optimizer": opt, "lr_scheduler": scheduler}

    def train_dataloader(self):
        return DataLoader(self.train_datasets, batch_size=self.train_batch_size, drop_last=True, shuffle=True)

    # def val_dataloader(self):
    #     return DataLoader(self.val_datasets, batch_size=self.val_batch_size, drop_last=True, shuffle=True)
