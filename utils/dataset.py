
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset


import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

class SegmentationDataset(Dataset):
    """
    Класс для того чтобы можно было совершать одинаковые преобразования
    над картинкой и маской
    """

    def __init__(self, data_path, mask_path, transform):
        """
        Args:
            data_path: путь до изображений.
            mask_path: путь до масок изображений.
            transform: трансформация для изображений.
        """
        self.data_path = data_path
        self.mask_path = mask_path
        self.transform = transform

        self.file_list = os.listdir(self.data_path)
        self.mask_list = os.listdir(self.mask_path)

        #Теперь отсортируем их, чтобы получить соответсвие.

        self.file_list.sort()
        self.mask_list.sort()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_filename = self.file_list[idx]
        mask_filename = self.mask_list[idx]
        
        image = cv2.imread(os.path.join(self.data_path, image_filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
        mask = cv2.imread(os.path.join(self.mask_path, mask_filename))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"][:,:,0]
        mask[mask != 0] = 1

        return transformed["image"], mask
