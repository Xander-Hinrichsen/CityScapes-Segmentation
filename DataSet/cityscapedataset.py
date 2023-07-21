import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image

os.listdir('../input/cityscapes/Cityspaces/images/train/aachen')
cityscapes_dir = '../input/cityscapes/Cityspaces'
image_dir = cityscapes_dir + '/images/train'
labels_dir = cityscapes_dir + '/gtFine/train'
val_image_dir = cityscapes_dir + '/images/val'
val_labels_dir = cityscapes_dir + '/gtFine/val'

cities = sorted(os.listdir(image_dir))

img_paths_list = []
mask_paths_list = []
for city in cities:
    city_imgs = sorted(os.listdir(image_dir + "/" + city))
    for j in range(len(city_imgs)):
        img_paths_list.append(image_dir + '/' + city + '/' + city_imgs[j])
        mask_paths_list.append(labels_dir + '/' + city + '/' + city_imgs[j][:len(city) + 15] + 'gtFine_labelTrainIds.png')

val_cities = sorted(os.listdir(val_image_dir))

val_img_paths_list = []
val_mask_paths_list = []
for val_city in val_cities:
    val_city_imgs = sorted(os.listdir(val_image_dir + "/" + val_city))
    for j in range(len(val_city_imgs)):
        val_img_paths_list.append(val_image_dir + '/' + val_city + '/' + val_city_imgs[j])
        val_mask_paths_list.append(val_labels_dir + '/' + val_city + '/' + val_city_imgs[j][:len(val_city) + 15] + 'gtFine_labelTrainIds.png')

class Dataset():
    def __init__(self, ds_type, im_size=(1024,2048), device=None, length=None, transform=False):
        if ds_type == 'train':
            X_paths=img_paths_list; y_paths=mask_paths_list
        elif ds_type == 'val':
            X_paths=val_img_paths_list; y_paths=val_mask_paths_list
        else:
            print('ds_type needs to be either "train" or "val"')
            
        assert len(X_paths) == len(y_paths)
        self.img_paths = X_paths
        self.mask_paths = y_paths
        ##this image transformation scales input to [0,1] so don't use on training mask
        self.to_tensor_x_only = torchvision.transforms.ToTensor()
        self.resizer = torchvision.transforms.Resize(im_size, interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        self.im_size = im_size
        self.length=length
        self.flipper = torchvision.transforms.RandomHorizontalFlip(p=1)
        self.normalizer = torchvision.transforms.Normalize((0.28693876, 0.32519343, 0.28395027), 
                                                           (0.17626978, 0.18112388, 0.17785645))
        self.to_transform = transform
    
    def __getitem__(self,i):
        to_flip = np.random.choice(2,1)
        to_flip = to_flip == 0
        #get the actual image - Xi
        with Image.open(self.img_paths[i]) as img:
            Xi = self.to_tensor_x_only(img)
            if self.to_transform:
                if to_flip:
                    Xi = self.flipper(Xi)
            Xi = self.resizer(Xi)
            if self.to_transform:
                Xi = self.normalizer(Xi)
        with Image.open(self.mask_paths[i]) as img2:
            yi_np = np.array(img2)
            ##misc class idx is stored as 255 - we only have 19 other classes 
            ##make misc class idx 19 instead (0 indexed obviously)
            yi_np[yi_np == 255] = 19
            ##convert to long tensor (tensor of torch.int datatypes)
            yi = torch.from_numpy(yi_np).long()
            yi = self.resizer(yi.reshape(1,1024,2048)).reshape(self.im_size[0],self.im_size[1]).long()
            if self.to_transform:
                if to_flip:
                    yi = self.flipper(yi)
            
        return Xi, yi
    
    def __len__(self):
        if self.length != None:
            return self.length
        return len(self.mask_paths)

# tds = Dataset('train', im_size=(512,1024), transform=True)

#fence might be wall!
##note - rider (bike rider) and person are pretty much the same class - should fix this - or maybe
##cross entropy loss isn't the best loss function to use here - have to test it out
##wall/fence not clear - as well as vegetation/terrain - is there terrain?
class CityScapeClasses():
    def __init__(self):
        self.classes = ['road', 'sidewalk', 'building', 'fence', 'pedestrian-railing', 'pole', 'traffic-light',
                       'traffic-sign', 'tree', 'vegetation', 'sky', 'person', 'rider', 'car', 'truck', 'bus',
                        'train', 'motorcycle', 'bicycle', 'misc']
