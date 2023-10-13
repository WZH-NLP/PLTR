import argparse
import json
import os
import random

from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.model.model import PLTRModel
from src.data_processing.dataset import NerDataset
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='bert-base-cased')
    parser.add_argument('--train_data_path', type=str, default='runs/drf_annotate_data')
    parser.add_argument('--val_data_path', type=str, default='NER_datasets/conll2003/dev.txt')
    parser.add_argument('--drf_save_path', type=str, default='runs/drf.json')
    parser.add_argument('--test_root_data_path', type=str, default='NER_datasets')
    parser.add_argument('--result_root_save_path', type=str, default='Results')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--label_list', type=str, nargs='+',
                        default=['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])
    parser.add_argument('--test_domain', type=str, nargs='+',
                        default=['conll2003', 'tech_news', 'ai', 'literature', 'music', 'politics', 'science'])
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--mask_num', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=256)
    parser.add_argument('--max_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--drf_constant', type=float, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--scheduler_type', type=str, default='linear')
    args = parser.parse_args()
    metric = load_metric("src/utils/my_seqeval.py")
    if args.seed == -1:
        seed = random.randint(1, 10000)
    else:
        seed = args.seed
    seed_everything(seed)
    model = PLTRModel(ckpt=args.ckpt,
                      hidden_dropout_prob=args.hidden_dropout_prob,
                      train_data_path=args.train_data_path,
                      val_data_path=args.val_data_path,
                      label_list=args.label_list,
                      mask_num=args.mask_num,
                      max_len=args.max_len,
                      drf_constant=args.drf_constant,
                      learning_rate=args.learning_rate,
                      train_batch_size=args.train_batch_size,
                      val_batch_size=args.val_batch_size,
                      metric=metric,
                      drf_path=args.drf_save_path,
                      max_epoch=args.max_epoch,
                      scheduler_type=args.scheduler_type
                      )
    if not os.path.exists(args.result_root_save_path):
        os.makedirs(args.result_root_save_path)
    if "roberta" in args.ckpt.lower():
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    # trainer = Trainer(max_epochs=args.max_epoch, precision=args.training_precision)
    trainer = Trainer(max_epochs=args.max_epoch, gpus=1, enable_checkpointing=False)
    print("parameters:\n",
          f"args.seed:{seed}\n",
          model.name, '\n',
          f"args.learning_rate:{args.learning_rate}\n",
          f"args.ckpt:{args.ckpt}\n",
          f"args.hidden_dropout_prob:{args.hidden_dropout_prob}\n",
          f"args.mask_num:{args.mask_num}\n",
          f"args.drf_constant:{args.drf_constant}\n",
          f"args.max_epoch:{args.max_epoch}\n",
          f"args.train_batch_size:{args.train_batch_size}\n",
          f"result_root_save_path:{args.result_root_save_path}")
    print("\n\n\n\n\n\n\n\n<<<----------Start training----------->>>")

    val_dataloaders = []
    test_dataloaders = []
    val_dataset = NerDataset('val', tokenizer, args.val_data_path, args.label_list, args.mask_num, args.max_len)
    val_dataloaders.append(DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=True))
    for domain in args.test_domain:
        dataset = NerDataset('test', tokenizer, args.test_root_data_path, args.label_list, args.mask_num, args.max_len,
                             domain)
        test_dataloaders.append(DataLoader(dataset, batch_size=args.val_batch_size, drop_last=True))
    trainer.fit(model,val_dataloaders=test_dataloaders)
    # trainer.test(model, dataloaders=test_dataloaders)
    file_path = os.path.join(args.result_root_save_path, "conclusion.txt")
    model.write_to_file(file_path)
    # trainer.fit(model)
    # f1 = {domain: 0 for domain in args.test_domain}
    # for domain in args.test_domain:
    #     print("\n\n\n\n\n\n\n\n<<<----------Now Evaluate:----------->>>", domain)
    #     dataset = NerDataset('test', tokenizer, args.test_root_data_path, args.label_list, args.mask_num, args.max_len,
    #                          domain)
    #     trainer.test(model, DataLoader(dataset, batch_size=args.val_batch_size, drop_last=True))
    #     name = domain + ".json"
    #     path = os.path.join(args.result_root_save_path, name)
    #     model.write_to_file(path)
    #     f1[domain] = model.get_current_f1_score()
    # file_path = os.path.join(args.result_root_save_path, "conclusion.txt")
    # total_score = 0
    # for v in f1.values():
    #     total_score += v
    # with open(file_path, "w") as f:
    #     print(f"total_score:{total_score}", file=f)
    #     print("parameters:\n",
    #           f"args.seed:{args.seed}\n",
    #           model.name, '\n',
    #           f"args.learning_rate:{args.learning_rate}\n",
    #           f"args.ckpt:{args.ckpt}\n",
    #           f"args.hidden_dropout_prob:{args.hidden_dropout_prob}\n",
    #           f"args.mask_num:{args.mask_num}\n",
    #           f"args.drf_constant:{args.drf_constant}\n",
    #           f"args.max_epoch:{args.max_epoch}\n",
    #           f"args.train_batch_size:{args.train_batch_size}\n",
    #           "No same drfs\n",
    #           file=f)
    #     print(json.dumps(f1), file=f)


if __name__ == '__main__':
    run()
