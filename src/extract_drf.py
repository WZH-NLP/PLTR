from src.data_processing import process_data
from sklearn.feature_extraction.text import CountVectorizer
from src.utils.ctfidf import CTFIDFVectorizer
from sklearn.metrics import mutual_info_score
import json
from transformers import AutoTokenizer
import src.annotate_prompt as annotate_prompt

DRF_SET_LOCATION = '../runs/drf_tf_idf'


def c_tf_idf(grouped_data, num_per_domain, total_class_num, rho) -> dict:
    """
    extract drf by cTfIdf
    :param grouped_data: pd.DataFrame(columns=['tokens', 'tags', 'class', 'sentence'])
    :param num_per_domain: nums of DRF
    :return: drf words per domain
    """
    docs_per_class = grouped_data[['class', 'sentence']].groupby(['class'], as_index=False).agg({'sentence': ' '.join})
    print(docs_per_class['sentence'])
    count_vectorizer = CountVectorizer().fit(docs_per_class['sentence'])
    count = count_vectorizer.transform(docs_per_class['sentence'])
    words = count_vectorizer.get_feature_names_out()
    print("extract DRF by C-TF-IDF")
    # Extract top 5 words
    count_array = count.toarray()
    print(count_array.shape)
    count_sum = count_array.sum(axis=0)
    ctfidf = CTFIDFVectorizer().fit_transform(count, n_samples=len(grouped_data)).toarray()
    words_per_domain = {}
    for class_num in range(0, total_class_num):  # calculate the rate of counts to determine the specific words
        words_in_one_class = []
        for word_id in ctfidf[class_num].argsort()[::-1]:
            if len(words_in_one_class) == num_per_domain:
                break
            if words[word_id].isnumeric():
                continue
            if count_sum[word_id] / count_array[class_num][word_id] < rho:
                words_in_one_class.append(words[word_id])
        words_per_domain[docs_per_class['class'][class_num]] = words_in_one_class

    # words_per_domain = {docs_per_class['class'][label]: [words[index] for index in ctfidf[label].argsort()[-num_per_domain:]] for
    #                    label in [10_mean2, 1, 2, 3]}
    print(words_per_domain)
    file_name = DRF_SET_LOCATION + '.json'
    print("writing to", file_name, "...")
    with open(file_name, 'w') as f:
        f.write(json.dumps(words_per_domain))
    return words_per_domain


def mutual_information(grouped_data, num_per_domain, total_class_num, drf_save_path, ckpt, rho) -> dict:
    """
    extract drf by mutual information
    :param rho: 3
    :param drf_save_path: save path
    :param total_class_num: 4
    :param grouped_data: pd.DataFrame(columns=['tokens', 'tags', 'class', 'sentence'])
    :param num_per_domain: nums of DRF
    :return: drf words per domain
    """
    #  get the counts of each words
    if "roberta" in ckpt.lower():
        tokenizer = AutoTokenizer.from_pretrained(ckpt, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(ckpt)
    drf_id_list = []
    docs_per_class = grouped_data[['class', 'sentence']].groupby(['class'], as_index=False).agg({'sentence': ' '.join})
    print(docs_per_class.info())
    count_vector = CountVectorizer(lowercase=False)
    count_fit = count_vector.fit_transform(docs_per_class['sentence'])
    count_vector_array = count_fit.toarray()
    # print(count_vector_array.shape)
    count_sum = count_vector_array.sum(axis=0)
    # print(count_vector.vocabulary_['the'])

    # get the total CountVectorizer all sentences * all words
    total_vector = CountVectorizer(binary=True, stop_words='english', lowercase=False)
    total_array = total_vector.fit_transform(grouped_data['sentence']).toarray()

    # drf dict of each class
    words_per_domain = {}
    for class_id in range(0, total_class_num):
        drfs_in_one_class = []  # drf in this class
        src_class_name = docs_per_class['class'][class_id]
        print("Now extracting DRF of class by MI:", src_class_name)
        mi_vector_of_src = []
        for i in range(0, grouped_data.shape[0]):
            if grouped_data.loc[i]['class'] == src_class_name:
                mi_vector_of_src.append(1)
            else:
                mi_vector_of_src.append(0)
        MI = {}
        for i in range(0, total_array.shape[1]):
            temp = mutual_info_score(total_array[:, i], mi_vector_of_src)
            MI[i] = temp
        # print(MI)
        MIs = sorted(MI.items(), key=lambda x: x[1], reverse=True)
        # print(MIs)
        for word_info in MIs:
            word_name = total_vector.get_feature_names_out()[word_info[0]]
            count_word_id = count_vector.vocabulary_[word_name]
            if len(drfs_in_one_class) == num_per_domain:
                break
            if count_vector_array[class_id][count_word_id] == 0 or any(ch.isdigit() for ch in word_name):
                continue
            if count_sum[count_word_id] / count_vector_array[class_id][count_word_id] < rho:
                word_id = tokenizer(word_name)["input_ids"][1]
                if word_id not in drf_id_list:
                    drf_id_list.append(word_id)
                    drfs_in_one_class.append(word_name)
        words_per_domain[src_class_name] = drfs_in_one_class
    print(words_per_domain)
    file_name = drf_save_path
    print("writing to", file_name, "...")
    with open(file_name, 'w') as f:
        f.write(json.dumps(words_per_domain))
    return words_per_domain


if __name__ == '__main__':
    ori_data = process_data.read_augmented_data("../runs/aug_all_data")
    # print(ori_data)
    grouped_data = process_data.group_data_by_tag(ori_data, "../runs/group_data")
    print("---",grouped_data.info())
    # print("grouped_data's shape", grouped_data.shape)
    # drf_by_ctfidf = c_tf_idf(grouped_data, 10, 4)
    drf_by_mi = mutual_information(grouped_data, 60, 4, "../runs/drf.json", "bert-base-cased", 3)
    # drf_by = c_tf_idf(grouped_data, 20, 4, 2)
    annotate_prompt.run1(40, "../runs/group_data", "../runs/drf.json", "../runs/drf_annotate_data",
                         ['MISC', 'PER', 'ORG', 'LOC'], "bert-base-cased")
