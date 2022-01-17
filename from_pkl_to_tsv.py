
import csv
import pickle

for domain in ["airline", "books", "dvd", "electronics", "kitchen"]:
    for mode in ["train", "dev"]:
        orig_path = f'blitzer_data/{domain}/{mode}'
        new_path = f'blitzer_data/{domain}/{mode}.tsv'

        with open(orig_path, 'rb') as f:
            (examples, labels) = pickle.load(f)

        with open(new_path, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['sentence', 'label'])
            for ex, lbl in zip(examples, labels):
                tsv_writer.writerow([ex, lbl])

    mode = "test"
    orig_path = f'blitzer_data/{domain}/{mode}'
    new_path = f'blitzer_data/{domain}/{mode}.tsv'
    new_path_labeled = f'blitzer_data/{domain}/{mode}-labeled.tsv'

    with open(orig_path, 'rb') as f:
        (examples, labels) = pickle.load(f)

    with open(new_path, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['index', 'sentence'])
        for ex_id, ex in enumerate(examples):
            tsv_writer.writerow([ex_id, ex])

    with open(new_path_labeled, 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['sentence', 'label'])
        for ex, lbl in zip(examples, labels):
            tsv_writer.writerow([ex, lbl])
