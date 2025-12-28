import os
import shutil
import random
import yaml
from pathlib import Path
from tqdm import tqdm

def create_client_data_splits(base_dir, output_dir, num_clients=2, val_split=0.1, test_split=0.1, seed=42):
    """
    Create client-specific data splits for federated learning.
    
    Args:
        base_dir (str): Base directory containing 'train', 'valid', 'test' folders
        output_dir (str): Directory to save client data
        num_clients (int): Number of clients
        val_split (float): Fraction of training data to use for validation
        test_split (float): Fraction of training data to use for testing
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    
    # Define paths
    train_img_dir = os.path.join(base_dir, 'train', 'images')
    train_label_dir = os.path.join(base_dir, 'train', 'labels')
    
    # Get list of training images
    train_images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(train_images)
    
    # Split into clients (simple random split for demonstration)
    # In a real scenario, you might want to implement non-IID splits
    client_splits = []
    split_size = len(train_images) // num_clients
    
    for i in range(num_clients):
        if i == num_clients - 1:
            client_splits.append(train_images[i * split_size:])
        else:
            client_splits.append(train_images[i * split_size:(i + 1) * split_size])
    
    # Create client directories and copy data
    for client_id, client_images in enumerate(client_splits):
        print(f"Preparing data for client {client_id + 1}...")
        
        # Create client directories
        client_dir = os.path.join(output_dir, f'client_{client_id + 1}')
        os.makedirs(os.path.join(client_dir, 'train', 'images'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'train', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'valid', 'images'), exist_ok=True)
        os.makedirs(os.path.join(client_dir, 'valid', 'labels'), exist_ok=True)
        
        # Split client data into train/val
        split_idx = int(len(client_images) * (1 - val_split))
        train_imgs = client_images[:split_idx]
        val_imgs = client_images[split_idx:]
        
        # Copy training data
        for img in tqdm(train_imgs, desc=f"Client {client_id + 1} - Copying training data"):
            # Copy image
            src_img = os.path.join(train_img_dir, img)
            dst_img = os.path.join(client_dir, 'train', 'images', img)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label
            label_name = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(train_label_dir, label_name)
            dst_label = os.path.join(client_dir, 'train', 'labels', label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
        
        # Copy validation data
        for img in tqdm(val_imgs, desc=f"Client {client_id + 1} - Copying validation data"):
            # Copy image
            src_img = os.path.join(train_img_dir, img)
            dst_img = os.path.join(client_dir, 'valid', 'images', img)
            shutil.copy2(src_img, dst_img)
            
            # Copy corresponding label
            label_name = os.path.splitext(img)[0] + '.txt'
            src_label = os.path.join(train_label_dir, label_name)
            dst_label = os.path.join(client_dir, 'valid', 'labels', label_name)
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
    
    print("\nData preparation complete!")
    print(f"Created data for {num_clients} clients in {output_dir}")

def create_data_yaml(dataset_dir, output_path, class_names):
    """
    Create a YAML file for YOLOv8 dataset configuration.
    
    Args:
        dataset_dir (str): Base directory of the dataset
        output_path (str): Path to save the YAML file
        class_names (list): List of class names
    """
    data = {
        'train': os.path.join(dataset_dir, 'train'),
        'val': os.path.join(dataset_dir, 'valid'),
        'test': os.path.join(dataset_dir, 'test'),
        'nc': len(class_names),
        'names': class_names
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created dataset configuration at {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare data for federated learning')
    parser.add_argument('--base-dir', type=str, default='606_train_152_val_12_test',
                       help='Base directory containing train/val/test folders')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Directory to save client data')
    parser.add_argument('--num-clients', type=int, default=2,
                       help='Number of clients for federated learning')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Fraction of training data to use for validation')
    parser.add_argument('--test-split', type=float, default=0.1,
                       help='Fraction of training data to use for testing')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create client data splits
    create_client_data_splits(
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed
    )
    
    # Create dataset YAML file
    class_names = ['blanket', 'call_point', 'detector', 'extinguisher']
    create_data_yaml(
        dataset_dir=args.base_dir,
        output_path=os.path.join(args.base_dir, 'data.yaml'),
        class_names=class_names
    )
