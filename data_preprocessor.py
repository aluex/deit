#####################3
from numpy import source
import torch
from torchvision import transforms
import argparse
import os
from PIL import Image

from einops import rearrange


DEFAULT_DATA_PATH = os.path.expanduser('~/datasets/ilsvrc2012')
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def get_args_parser():
    parser = argparse.ArgumentParser('Data Transforms', add_help=False)
    parser.add_argument('--data', default=DEFAULT_DATA_PATH, type=str)
    parser.add_argument('--transform', default='none', type=str, choices=['uspk', 'none'])
    parser.add_argument('--patch-size', default=16, type=int)
    parser.add_argument('--channels', default=3, type=int)
    parser.add_argument('--image-size', default=224, type=int)
    return parser

def is_valid_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)

def perform_transform(filepath, source_dir, target_dir, rand_matrix, args):
    target_path = filepath.replace(source_dir, target_dir)
    with open(filepath, 'rb') as f:
        img = transforms.ToTensor()(transforms.RandomCrop((args.image_size, args.image_size))(transforms.Resize(args.image_size)(Image.open(f).convert('RGB'))))
        x = rearrange(img, 'c (h p1) (w p2) -> (h w) (c p1 p2) 1', p1=args.patch_size, p2=args.patch_size, c=args.channels)
        x = torch.matmul(rand_matrix, x)
        x = rearrange(x, '(h w) (c p1 p2) 1 -> c (h p1) (w p2)', h=args.image_size // args.patch_size, p1=args.patch_size, p2=args.patch_size, c=args.channels)
        img = transforms.ToPILImage()(x)
        img.save(target_path)

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    patch_num = args.image_size // args.patch_size
    rand_matrix = torch.randn(size=(patch_num ** 2, args.channels * args.patch_size**2, args.channels * args.patch_size**2))
    target_data_directory = args.data + '_' + args.transform + str(patch_num)
    print('Writing to', target_data_directory)
    if not os.path.exists(target_data_directory):
        os.makedirs(target_data_directory)
    classes = [d.name for d in os.scandir(args.data) if d.is_dir()]
    for target_class in classes:
        from_dir = os.path.join(args.data, target_class)
        target_dir = os.path.join(target_data_directory, target_class)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        
        for root, _, fnames in sorted(os.walk(from_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    perform_transform(path, from_dir, target_dir, rand_matrix, args)
