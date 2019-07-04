import pandas as pd
import json
from pathlib import Path
from collections import namedtuple


def generate_metadata(fold_id):
    # Sanity check.
    if fold_id not in [1, 2, 3]:
        raise ValueError

    # Read the text files which indicate which videos are in the training and testing sets.
    splits_dir = Path(r'E:\hmdb51_org\test_train_splits\testTrainMulti_7030_splits')
    split_files = list(splits_dir.glob(f'*_split{fold_id}.txt'))

    # Assign a numerical value to each class (if not already existing).
    class_mapping_path = Path('./hmdb_class_mapping.json')
    if not class_mapping_path.exists():
        class_mapping = [x.stem.split('_test_split')[0] for x in split_files]
        class_mapping = dict(zip(class_mapping, range(len(class_mapping))))
        json.dump(class_mapping, class_mapping_path.open('w'))
        print('Created class mapping file (since one does not already exist).')
    else:
        class_mapping = json.load(class_mapping_path.open('r'))

    # Create a named tuple to represent each data sample.
    DataSample = namedtuple('DataSample', ['name', 'category_name', 'category', 'training_split'])

    # Iterate over each category and specify which files belong to the training and testing splits.
    data = []
    for split_file in split_files:
        category_name = split_file.stem.split('_test_split')[0]

        split_file = split_file.open('r').readlines()
        for line in split_file:
            name, split_type = line.split()

            # Store the information associated with each file in the dataset.
            if split_type == '0':
                continue
            else:
                data_sample = DataSample(name=name.replace('.avi', ''),
                                         category_name=category_name,
                                         category=class_mapping[category_name],
                                         training_split=split_type == '1'
                                         )
                data.append(data_sample)

    # Save to file.
    data = pd.DataFrame(data)
    data.to_csv(f'hmdb51_metadata_split_{fold_id}.csv')
    print(f'Saved metadata to: [hmdb51_metadata_split_{fold_id}.csv]')


def main():
    for i in [1, 2, 3]:
        generate_metadata(i)


if __name__ == '__main__':
    main()
