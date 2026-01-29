import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader as dataloader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from PIL import Image, ImageOps
import copy
import pandas as pd
from cub200_dataset import CUB200
from cub200_trainer import ModelTrainer

batch_size = 64
num_epochs = 25
learning_rate = 1e-4

data_path = 'data/cub200/'

image_size = 128

start_from_checkpoint = False
save_dir = 'data/Models'
model_name = 'ResNet34_CUB200_V3'

class BboxIOU(nn.Module):
    def xyhw_to_xyxy(self, bbox):
        '''Converts xy-height-width bbox to xy-coordinates.'''
        new_bbox = torch.cat((bbox[:,0:1],
                              bbox[:,1:2],
                              bbox[:, 2:3] + bbox[:, 0:1],
                              bbox[:, 3:4] + bbox[:, 1:2]), 1)
        return new_bbox

    def bb_IOU(self, pred_xyhw, target_xyhw):
        '''Calculates the IOU between two bounding boxes.'''
        pred_xyxy = self.xyhw_to_xyxy(pred_xyhw)
        target_xyxy = self.xyhw_to_xyxy(target_xyhw)

        xA = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0]).unsqueeze(1)
        yA = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1]).unsqueeze(1)
        xB = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2]).unsqueeze(1)
        yB = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3]).unsqueeze(1)

        x_len = F.relu(xB - xA)
        y_len = F.relu(yB - yA)
        interArea = x_len * y_len

        area1 = pred_xyhw[:, 2:3] * pred_xyhw[:, 3:4]
        area2 = target_xyhw[:, 2:3] * target_xyhw[:, 3:4]

        iou = interArea / (area1 + area2 - interArea + 1e-8)

        return iou

    def forward(self, predictions, data):
        pred_bbox = torch.sigmoid(predictions[:, :4])
        target_bbox = data[1].to(pred_bbox.device)

        return self.bb_IOU(pred_bbox, target_bbox)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([A.SmallestMaxSize(max_size=image_size),
                                 A.RandomCrop(height=image_size, width=image_size),
                                 A.HorizontalFlip(p=0.5),
                                 A.Affine(p=0.5),
                                 A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                                 A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                                 A.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
                                 ToTensorV2()],
                                bbox_params=A.BboxParams(format='coco',
                                                         min_area=0, min_visibility=0.0,
                                                         label_fields=['class_labels']))

    transform = A.Compose([A.SmallestMaxSize(max_size=image_size),
                           A.RandomCrop(height=image_size, width=image_size),
                           A.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]),
                           ToTensorV2()],
                          bbox_params=A.BboxParams(format='coco',
                                                   min_area=0, min_visibility=0.0,
                                                   label_fields=['class_labels']))

    train_data = CUB200(data_path, image_size, train_transform, test_train=0)
    test_data = CUB200(data_path, image_size, transform, test_train=1)

    validation_split = 0.9

    n_train_examples = int(len(train_data)*(validation_split))
    n_valid_examples = len(train_data) - n_train_examples
    train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples], generator=torch.Generator().manual_seed(42))

    res_net = models.resnet34(weights='IMAGENET1K_V1')

    model_trainer = ModelTrainer(model=res_net.to(device), output_size=4, device=device,
                                 loss_fun=nn.BCEWithLogitsLoss(), batch_size=batch_size,
                                 learning_rate=learning_rate, save_dir=save_dir, model_name=model_name,
                                 eval_metric=BboxIOU(), start_from_checkpoint=start_from_checkpoint)

    model_trainer.set_data(train_set=train_data, test_set=test_data, val_set=valid_data)

    model_trainer.set_lr_schedule(optim.lr_scheduler.StepLR(model_trainer.optimizer, step_size=1, gamma=0.95))

    plt.figure(figsize = (20,10))
    images, bbox, labels = next(iter(model_trainer.test_loader))
    out = torchvision.utils.make_grid(images, normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.title('Subset of training images')
    plt.show()


    example_idx = 3
    ex_img = images[example_idx]
    ex_label = bbox[example_idx].unsqueeze(0) * image_size
    ex_label[:, 2] += ex_label[:, 0]
    ex_label[:, 3] += ex_label[:, 1]

    img_out = (((ex_img - ex_img.min())/(ex_img.max() - ex_img.min())) * 255).to(torch.uint8)
    img_box = torchvision.utils.draw_bounding_boxes(img_out, ex_label, colors=(0, 255, 0))


    plt.figure(figsize = (5,5))
    out = torchvision.utils.make_grid(img_box.unsqueeze(0).float(), normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.title('Example image with true bounding box')
    plt.show()


    model_trainer.run_training(num_epochs=num_epochs)

    print(f'The highest validation IoU was {model_trainer.best_valid_acc:.2f}')

    plt.figure(figsize=[10, 5])
    train_x = np.linspace(0, num_epochs, len(model_trainer.train_loss_logger))
    plt.plot(train_x, model_trainer.train_loss_logger, label='Train')
    plt.title('Training Loss')
    plt.show()

    example_indx = 50
    ex_img = images[example_indx]
    img_out = (((ex_img - ex_img.min()) / (ex_img.max() - ex_img.min())) * 255).to(torch.uint8)

    real_label = bbox[example_indx].unsqueeze(0) * image_size
    real_label[:, 2] += real_label[:, 0]
    real_label[:, 3] += real_label[:, 1]

    # Creates model predictions for bounding boxes
    model_trainer.eval()
    with torch.no_grad():
        pred_out = torch.sigmoid(model_trainer(ex_img.unsqueeze(0).to(device)))
        pred_label = (pred_out * image_size).cpu()
        pred_label[:, 2] += pred_label[:, 0]
        pred_label[:, 3] += pred_label[:, 1]

    # Adds the bounding boxes to the image
    img_box = torchvision.utils.draw_bounding_boxes(img_out, real_label, colors=(0, 255, 0))
    img_box = torchvision.utils.draw_bounding_boxes(img_box, pred_label, colors=(255, 0, 0))

    plt.figure(figsize=(5, 5))
    out = torchvision.utils.make_grid(img_box.unsqueeze(0).float(), normalize=True)
    plt.imshow(out.numpy().transpose((1, 2, 0)))
    plt.title('Example image with predicted bounding box')
    plt.show()

    plt.figure(figsize=(10, 5))
    train_x = np.linspace(0, num_epochs, len(model_trainer.train_acc_logger))
    plt.plot(train_x, model_trainer.train_acc_logger, c="y")
    valid_x = np.linspace(0, num_epochs, len(model_trainer.val_acc_logger))
    plt.plot(valid_x, model_trainer.val_acc_logger, c="k")

    plt.title("Average IoU")
    plt.legend(["Training IoU", "Validation IoU"])
    plt.show()

    test_acc = model_trainer.evaluate_model(train_test_val="test")
    print(f"The Test Average IoU is: {test_acc:.2f}")


if __name__ == '__main__':
    main()

