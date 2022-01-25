from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import argparse
import pickle
import os
import spacy
nlp = spacy.load('en_core_web_lg')


def GetTopNMI(n, X, target):
    MI = []
    length = X.shape[1]

    for i in range(length):
        temp = mutual_info_score(X[:, i], target)
        MI.append(temp)
    MIs = sorted(range(len(MI)), key=lambda i: MI[i])[-n:]
    return MIs, MI


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def getTopCounts(X, k):
    print(f"X type = {type(X)}")
    X_sum = np.sum(X, axis=0)
    print(f"X_sum = {X_sum.shape}")
    top_count = X_sum.argsort()[-k:][::-1]
    return top_count


def getCounts(X,i):
    return (sum(X[:,i]))


def preproc(pivot_num, src, dest, src_threshold=20, trg_threshold=50, n_gram=(1,1)):
    """find pivots from source and data domains with mutual information
    Parameters:
    pivot_num (list of int or int): number of pivots to find
    src (str): source domain name
    dest (str): target domain name
    src_threshold (int): minimal appearances of the pivot in source domains
    trg_threshold (int): minimal appearances of the pivot in target domains
    n_gram (tuple of integers): n_grams to include in pivots selection (min, max), default is 1 grams only
    Returns:
    list of pivots
   """

    base_path = os.getcwd() + os.sep
    pivotsCounts = []

    src_path = "blitzer_data/" + src + os.sep
    with open(src_path + "train", 'rb') as f:
        (train, train_labels) = pickle.load(f)
    with open(src_path + "unlabeled", 'rb') as f:
        source = pickle.load(f)

    target = []
    for trg in dest:
        if trg == src:
            continue
        # gets all the train and test for pivot classification
        dest_path = "blitzer_data/" + trg + os.sep
        with open(dest_path + "unlabeled", 'rb') as f:
            tmp_target = pickle.load(f)
        target += tmp_target

    source = source + train
    unlabeled = source + target

    # sets x train matrix for classification
    print('starting bigram_vectorizer for train data...')
    bigram_vectorizer = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=5,
                                        binary=True, stop_words='english')
    X_2_train = bigram_vectorizer.fit_transform(train).toarray()
    print('Done!')

    print('starting bigram_vectorizer for unlabled data...')
    bigram_vectorizer_unlabeled = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b',
                                                  min_df=src_threshold+trg_threshold, binary=True, stop_words='english')
    bigram_vectorizer_unlabeled.fit_transform(unlabeled).toarray()

    print('Done!')

    print('starting bigram_vectorizer for source data...')
    bigram_vectorizer_source = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=src_threshold,
                                               binary=True, stop_words='english')
    X_2_train_source = bigram_vectorizer_source.fit_transform(source).toarray()
    print('Done!')

    print('starting bigram_vectorizer for target data...')
    bigram_vectorizer_labels = CountVectorizer(ngram_range=n_gram, token_pattern=r'\b\w+\b', min_df=trg_threshold,
                                               binary=True, stop_words='english')
    X_2_train_labels = bigram_vectorizer_labels.fit_transform(target).toarray()
    print('Done!')

    # get a sorted list of pivots with respect to the MI with the label
    print('starting calculating MI...')
    MIsorted, RMI = GetTopNMI(2000, X_2_train, train_labels)
    MIsorted.reverse()

    names, positive_pivots, negative_pivots = [], [], []
    for i, MI_word in enumerate(MIsorted):
        name = bigram_vectorizer.get_feature_names()[MI_word]
        if len(name) > 2 and not hasNumbers(name):
            tokens = nlp("good bad" + " " + name)
            token1, token2, token3 = tokens[0], tokens[1], tokens[2]
            pos_sim, neg_sim = token1.similarity(token3), token2.similarity(token3)

            s_count = getCounts(X_2_train_source, bigram_vectorizer_source.get_feature_names().index(
                name)) if name in bigram_vectorizer_source.get_feature_names() else 0
            t_count = getCounts(X_2_train_labels, bigram_vectorizer_labels.get_feature_names().index(
                name)) if name in bigram_vectorizer_labels.get_feature_names() else 0

            # pivot must meet 2 conditions, to have high MI with the label and appear at least pivot_min_st times in the
            # source and target domains
            if s_count >= src_threshold and t_count >= trg_threshold and abs(pos_sim - neg_sim) > 0.1:
                if pos_sim - neg_sim > 0.1 and len(positive_pivots) < pivot_num:
                    names.append(name)
                    positive_pivots.append('Ġ'+name)
                elif neg_sim - pos_sim > 0.1 and len(negative_pivots) < pivot_num:
                    names.append(name)
                    negative_pivots.append('Ġ'+name)
                pivotsCounts.append(bigram_vectorizer_unlabeled.get_feature_names().index(name))

            if len(negative_pivots) == pivot_num and len(positive_pivots) == pivot_num:
                break

    print(f"pivots = {names}")
    print(f"    positive = {positive_pivots}")
    print(f"    negative = {negative_pivots}")
    filename = base_path + 'label_tokens/mi/' + src
    with open(filename, 'wb') as f:
        pickle.dump((negative_pivots, positive_pivots), f)

    return names, positive_pivots, negative_pivots


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pivot_num",
                        default=3,
                        type=int,
                        help="The number of selected pivots")
    parser.add_argument("--src_threshold",
                        default=20,
                        type=int,
                        help="Minimum counts of pivots in src")
    parser.add_argument("--trg_threshold",
                        default=50,
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
                        default='unigram',
                        type=str,
                        help="N_gram length.")

    args = parser.parse_args()

    if args.n_gram == "bigram":
        n_gram = (1, 2)
    elif args.n_gram == "unigram":
        n_gram = (1, 1)
    else:
        print("This code does not soppurt this type of n_gram")
        exit(0)

    _ = preproc(pivot_num=args.pivot_num, src=args.src, dest=args.dest, src_threshold=args.src_threshold,
                trg_threshold=args.trg_threshold, n_gram=n_gram)


if __name__ == "__main__":
    main()
