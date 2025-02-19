import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(data_dir, test_size=0.2, random_state=None):
    # Create training and testing directories
    train_dir = os.path.join(data_dir, 'training')
    test_dir = os.path.join(data_dir, 'testing')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Iterate over each class directory
    for class_name in os.listdir(data_dir):
        if class_name in ['training', 'testing']:
            continue  # Skip the training and testing directories

        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Get all files in the class directory
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        if not files:
            print(f"No files found in {class_dir}, skipping.")
            continue

        # Split the files into training and testing sets
        train_files, test_files = train_test_split(
            files,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )

        # Create target directories
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Copy files to respective directories
        for f in train_files:
            shutil.move(os.path.join(class_dir, f), train_class_dir)
        for f in test_files:
            shutil.move(os.path.join(class_dir, f), test_class_dir)

        print(f"Class {class_name}: {len(train_files)} training, {len(test_files)} testing samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split dataset into training and testing sets.')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to the dataset directory (default: data)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to include in the test split (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=None,
                        help='Seed for random number generator (default: None)')

    args = parser.parse_args()

    split_dataset(
        args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state
    )

    print("Dataset splitting completed successfully.")