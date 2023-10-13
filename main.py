import argparse
import src.data_processing.data_augmenter as da
import src.extract_drf as drf
import src.data_processing.process_data as process_data
import src.annotate_prompt as annotate_prompt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # augmentation
    parser.add_argument('--ckpt', type=str, default='bert-base-cased')
    parser.add_argument('--aug_batch_size', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--output_pad_token', type=str, default='O')
    parser.add_argument('--sampled_trainset_save_path', type=str, default='runs/ori_train_dataset')
    parser.add_argument('--ori_model_save_path', type=str, default='runs/ori_model')
    parser.add_argument('--semi_data_save_path', type=str, default='runs/aug_semi_data')
    parser.add_argument('--counter_data_save_path', type=str, default='runs/aug_counter_data')
    parser.add_argument('--all_data_save_path', type=str, default='runs/aug_all_data')
    parser.add_argument('--all_train_dataset', type=str, default=None)
    parser.add_argument('--semi_aug_ratio', type=int, default=-1)
    parser.add_argument('--counter_aug_ratio', type=int, default=-1)
    # extracting drfs
    parser.add_argument('--group_data_save_path', type=str, default='runs/group_data')
    parser.add_argument('--drf_save_path', type=str, default='runs/drf.json')
    parser.add_argument('--drf_num_per_domain', type=int, default=60)
    parser.add_argument('--drf_rho', type=int, default=3)  # metric about domain_specific
    parser.add_argument('--class_list', type=str, nargs='+', default=['MISC', 'PER', 'ORG', 'LOC'])  # MISC PER LOC ORG
    parser.add_argument('--label_list', type=str, nargs='+',
                        default=['O', 'B-MISC', 'I-MISC', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC'])
    # annotating prompt
    parser.add_argument('--ann_drf_num', type=int, default=40)
    parser.add_argument('--ann_save_path', type=str, default='runs/drf_annotate_data')

    args = parser.parse_args()
    data_augmenter_args = dict(ckpt=args.ckpt,
                               batch_size=args.aug_batch_size,
                               output_pad_token=args.output_pad_token,
                               num_samples=args.num_samples,  # numbers of ori train set
                               all_train_dataset=args.all_train_dataset,
                               label_list=args.label_list,  # all ner labels
                               sampled_trainset_save_path=args.sampled_trainset_save_path,
                               ori_model_save_path=args.ori_model_save_path,
                               semi_ratio=args.semi_aug_ratio,
                               counter_ratio=args.counter_aug_ratio)
    data_augmenter = da.DataAugmenter(**data_augmenter_args)
    data_augmenter.run(args.semi_data_save_path, args.counter_data_save_path, args.all_data_save_path)
    print("\n\n\n\n\n\n\n\n<<<----------reading augmented data----------->>>")
    ori_data = process_data.read_augmented_data(args.all_data_save_path)
    print("\n\n\n\n\n\n\n\n<<<----------extracting drfs----------->>>")
    grouped_data = process_data.group_data_by_tag(ori_data, args.group_data_save_path)
    drf_by_mi = drf.mutual_information(grouped_data, args.drf_num_per_domain, len(args.class_list), args.drf_save_path,
                                       args.ckpt, args.drf_rho)
    print("\n\n\n\n\n\n\n\n<<<----------annotating prompt----------->>>")
    annotate_prompt.run1(args.ann_drf_num, args.group_data_save_path, args.drf_save_path, args.ann_save_path,
                         args.class_list, args.ckpt)
