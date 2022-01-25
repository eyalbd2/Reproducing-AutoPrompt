from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import argparse
import pickle
import os
from pathlib import Path


def GetTopNMI(n, X, target):
    MI = []
    length = X.shape[1]

    for i in range(length):
        temp = mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs, MI


def getTopCounts(X, k):
    print(f"X type = {type(X)}")
    X_sum = np.sum(X, axis=0)
    print(f"X_sum = {X_sum.shape}")
    top_count = X_sum.argsort()[-k:][::-1]
    return top_count


def getCounts(X,i):
    return (sum(X[:,i]))


def preproc(src, dest, src_threshold=20, trg_threshold=50, n_gram=(1,1), label_tokens_path=None):
    """find pivots from source and data domains with mutual information
    Parameters:
    src (str): source domain name
    dest (str): target domain name
    src_threshold (int): minimal appearances of the pivot in source domains
    trg_threshold (int): minimal appearances of the pivot in target domains
    n_gram (tuple of integers): n_grams to include in pivots selection (min, max), default is 1 grams only
    Returns:
    list of pivots
   """

    base_path = os.getcwd() + os.sep
    trigger_counts = []

    with open(label_tokens_path, 'rb') as f:
        label_token_list = pickle.load(f)
    neg_label_tokens, pos_label_tokens = label_token_list[0], label_token_list[1]
    neg_label_tokens = [ng[1:] for ng in neg_label_tokens]  # omit 'Ġ' when using roberta-base.
    pos_label_tokens = [ps[1:] for ps in pos_label_tokens]  # omit 'Ġ' when using roberta-base.

    src_path = "blitzer_data/" + src + os.sep
    source, source_labels = [], []
    with open(src_path + "unlabeled", 'rb') as f:
        tmp_source = pickle.load(f)
    for txt in tmp_source:
        words = txt.split()
        for word_id in range(len(words[:-4])):
            cur_str = " ".join(words[int(word_id):int(word_id+3)])
            source.append(cur_str)
            if words[int(word_id+3)] in neg_label_tokens + pos_label_tokens:
                source_labels.append(1)
            else:
                source_labels.append(0)

    target, target_labels = [], []
    for trg in dest:
        if trg == src:
            continue
        # gets all the train and test for pivot classification
        dest_path = "blitzer_data/" + trg + os.sep
        with open(dest_path + "unlabeled", 'rb') as f:
            tmp_target = pickle.load(f)
        tmp_target = tmp_target[:5000]
        for txt in tmp_target:
            words = txt.split()
            for word_id in range(len(words[:-4])):
                cur_str = " ".join(words[int(word_id):int(word_id + 3)])
                target.append(cur_str)
                if words[int(word_id + 3)] in neg_label_tokens + pos_label_tokens:
                    target_labels.append(1)
                else:
                    target_labels.append(0)

    unlabeled = source + target
    labels = source_labels + target_labels

    print(f"len unlabeled = {len(unlabeled)}, {len(labels)}")

    # negative_text, positive_text = [], []
    # for txt in unlabeled:
    #     neg_counts, pos_counts = 0, 0
    #     for token in neg_label_tokens:
    #         if token in txt:
    #             neg_counts += 1
    #     for token in pos_label_tokens:
    #         if token in txt:
    #             pos_counts += 1
    #     if neg_counts > pos_counts:
    #         negative_text.append(txt)
    #     elif pos_counts > neg_counts:
    #         positive_text.append(txt)
    #     else:
    #         negative_text.append(txt)
    #         positive_text.append(txt)

    # sets x train matrix for classification
    print('starting vectorizer for unlabled data...')
    vectorizer_unlabeled = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b',
                                           min_df=src_threshold+trg_threshold, binary=True)
    x_unlabeled = vectorizer_unlabeled.fit_transform(unlabeled).toarray()
    print('Done!')

    print('starting vectorizer for source data...')
    vectorizer_source = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=src_threshold,
                                        binary=True)
    x_source = vectorizer_source.fit_transform(source).toarray()
    print('Done!')

    print('starting vectorizer for target data...')
    vectorizer_target = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=trg_threshold,
                                        binary=True)
    x_target = vectorizer_target.fit_transform(target).toarray()
    print('Done!')

    # print('starting vectorizer for negative data...')
    # vectorizer_neg = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=trg_threshold,
    #                                  binary=True)
    # x_negative = vectorizer_neg.fit_transform(negative_text).toarray()
    # print('Done!')

    # print('starting vectorizer for positive data...')
    # vectorizer_pos = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=trg_threshold,
    #                                  binary=True)
    # x_positive = vectorizer_pos.fit_transform(positive_text).toarray()
    # print('Done!')

    # get a sorted list of pivots with respect to the MI with the label
    print('starting MI calculation...')
    MIsorted, RMI = GetTopNMI(2000, x_unlabeled, labels)
    MIsorted.reverse()

    triggers = []
    for i, MI_word in enumerate(MIsorted):
        trigger = vectorizer_unlabeled.get_feature_names()[MI_word]
        s_count = getCounts(x_source, vectorizer_source.get_feature_names().index(
            trigger)) if trigger in vectorizer_source.get_feature_names() else 0
        t_count = getCounts(x_target, vectorizer_target.get_feature_names().index(
            trigger)) if trigger in vectorizer_target.get_feature_names() else 0
        # n_count = getCounts(x_negative, vectorizer_neg.get_feature_names().index(
        #     trigger)) if trigger in vectorizer_neg.get_feature_names() else 0
        # p_count = getCounts(x_positive, vectorizer_pos.get_feature_names().index(
        #     trigger)) if trigger in vectorizer_pos.get_feature_names() else 0

        # if s_count >= src_threshold and t_count >= trg_threshold and n_count >= trg_threshold and p_count >= trg_threshold:
        if s_count >= src_threshold and t_count >= trg_threshold:
            triggers.append(trigger)
            trigger_counts.append(vectorizer_unlabeled.get_feature_names().index(trigger))

    print(f"triggers = {triggers}")
    filename = base_path + 'trigger_tokens/mi/' + src
    with open(filename, 'wb') as f:
        pickle.dump(triggers, f)

    return triggers


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_threshold",
                        default=100,
                        type=int,
                        help="Minimum counts of pivots in src")
    parser.add_argument("--trg_threshold",
                        default=200,
                        type=int,
                        help="Minimum counts of pivots in dest")
    parser.add_argument("--src",
                        default='kitchen',
                        type=str,
                        help="Source domain.")
    parser.add_argument("--dest",
                        default=['airline', 'books', 'dvd', 'electronics', 'kitchen'],
                        type=list,
                        help="Destination domain.")
    parser.add_argument("--n_gram",
                        default='trigram',
                        type=str,
                        help="N_gram length.")
    parser.add_argument("--label_token_path",
                        default=None,
                        type=Path,
                        help="The path the the label tokens map.")

    args = parser.parse_args()

    if args.n_gram == "trigram":
        n_gram = (3, 3)
    elif args.n_gram == "bigram":
        n_gram = (2, 2)
    elif args.n_gram == "unigram":
        n_gram = (1, 1)
    else:
        print("This code does not soppurt this type of n_gram")
        exit(0)

    _ = preproc(src=args.src, dest=args.dest, src_threshold=args.src_threshold, trg_threshold=args.trg_threshold,
                n_gram=n_gram, label_tokens_path=args.label_token_path)


if __name__ == "__main__":
    main()
