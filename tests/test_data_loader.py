import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

from semantic_bac_segment.src.semantic_bac_segment.data_loader import BacSegmentDataset, collate_fn
from torch.utils.data import DataLoader


SEM_train = BacSegmentDataset(
    './data/train/source', './data/train/mask', mode = 'train')

imag_1 = SEM_train.__getitem__(0)

def test_data_loader():
    # Set up the dataset and data loader
    image_dir = './data/train/source'
    mask_dir = './data/train/mask'
    dataset = BacSegmentDataset(image_dir, mask_dir, mode='train', patch_size=512)
    batch_size = 2
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Check the number of batches
    expected_num_batches = len(dataset) // batch_size

    print(f'dataset len: {len(dataset)}')
    print(f'batch_size len: {batch_size}')
    print(f'expected_num_batches: {expected_num_batches}')

    for batch_idx, (images, masks) in enumerate(data_loader):
        print(f'batch_idx: {batch_idx}')
        print(f'images: {images.shape}')
        print(f'masks: {masks.shape}')

test_data_loader()