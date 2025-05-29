import shutil
import kagglehub
from Dataset import *

if __name__ == "__main__":
    # Dataset: Download latest version
    dataset_path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")

    # print("Path to dataset files:", dataset_path)

    # Dataset: For some reason, after extracting the dataset, 
    # there will always be 2 folders with almost identical name "Garbage classification" 
    # which contain the exact same image dataset.
    # Removing one of them would save space and eliminate confusion.
    dir_to_rm = f"{dataset_path}/garbage classification/"
    if os.path.isdir(dir_to_rm):
        shutil.rmtree(dir_to_rm)

    # Initialize data loader
    data_loader = TrashDataLoader(
        data_dir=dataset_path,
        batch_size=32,
        num_workers=4,
        image_size=224,
        augment_training=True
    )
    
    # # Get data loaders
    # train_loader = data_loader.get_train_loader()
    # val_loader = data_loader.get_val_loader()
    # test_loader = data_loader.get_test_loader()
    
    # print(f"Number of classes: {data_loader.get_num_classes()}")
    # print(f"Class names: {data_loader.get_class_names()}")
    
    # # Example: Load single image for inference
    # try:
    #     single_loader = data_loader.get_single_image_loader("path/to/single/image.jpg")
    #     print("Single image loader created successfully")
    # except FileNotFoundError as e:
    #     print(f"Error: {e}")
    
    # # Visualize a batch (uncomment to use)
    # # data_loader.visualize_batch(train_loader)