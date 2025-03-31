import os
import random
import shutil

# Path to dataset_images folder where all action subfolders currently are
DATASET_DIR = "dataset_images"

# create new 'train', 'val', and 'test' folders inside dataset_images
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Ratios for splitting
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15 
TEST_RATIO  = 0.15 

def main():
    # Make new train/val/test directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    # Loop over each action class folder
    for action_label in os.listdir(DATASET_DIR):
        # skip 'train', 'val', 'test' if they exist
        if action_label in ["train", "val", "test"]:
            continue

        action_path = os.path.join(DATASET_DIR, action_label)
        if not os.path.isdir(action_path):
            continue
        
        # List all image files in that action folder
        img_files = [f for f in os.listdir(action_path) if f.endswith(".jpg")]
        random.shuffle(img_files)

        total_imgs = len(img_files)
        train_count = int(total_imgs * TRAIN_RATIO)
        val_count   = int(total_imgs * VAL_RATIO)
        # test_count  = total_imgs - (train_count + val_count)

        train_files = img_files[:train_count]
        val_files   = img_files[train_count : train_count + val_count]
        test_files  = img_files[train_count + val_count :]

        # Create action subfolder inside train, val, test
        action_train_dir = os.path.join(TRAIN_DIR, action_label)
        action_val_dir   = os.path.join(VAL_DIR,   action_label)
        action_test_dir  = os.path.join(TEST_DIR,  action_label)
        os.makedirs(action_train_dir, exist_ok=True)
        os.makedirs(action_val_dir,   exist_ok=True)
        os.makedirs(action_test_dir,  exist_ok=True)
        
        # Move the files to their respective folders
        for f in train_files:
            src_path = os.path.join(action_path, f)
            dst_path = os.path.join(action_train_dir, f)
            shutil.move(src_path, dst_path)
        
        for f in val_files:
            src_path = os.path.join(action_path, f)
            dst_path = os.path.join(action_val_dir, f)
            shutil.move(src_path, dst_path)

        for f in test_files:
            src_path = os.path.join(action_path, f)
            dst_path = os.path.join(action_test_dir, f)
            shutil.move(src_path, dst_path)


    print("Dataset split into train/val/test complete.")

if __name__ == "__main__":
    main()
