import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import os
from PIL import Image

###We are assuming that we are making a prediction on one image at a time - because that's how real time predictions work

def iou_fn(map1, map2):
    intersection = (map1&map2).float().sum()
    union = (map1 | map2).float().sum() + 1e-6
    return intersection/union

def miou(preds, labels, num_classes=20):
    miou = 0 
    _, pred_idxs = torch.max(preds, dim=1)
    for i in range(preds.shape[0]):
        pred_map = pred_idxs[i]
        truth_map = labels[i]
        ious=[]
        for j in range(num_classes):
            pred_mapt = pred_map == j
            truth_mapt = truth_map == j
            iou = iou_fn(pred_mapt, truth_mapt)
            ious.append(iou)
        miou += torch.mean(torch.tensor(ious)).item()
    return miou / preds.shape[0]

def pixelwiseacc(preds, labels):
    _, preds_idxs = torch.max(preds, dim=1)
    acc = (torch.sum(preds_idxs==labels) / preds.shape[0]) / (preds_idxs.shape[1]*preds_idxs.shape[2])
    return acc.item()

def make_pred(model, Xi, img_shape):
    pred = model(Xi.reshape(-1,Xi.shape[0], Xi.shape[1], Xi.shape[2])).reshape(20, img_shape[0], img_shape[1])
    _, idxs = torch.max(pred, dim=0)
    return idxs

def colorify(idxs, img_shape, figsize=(20,20), plot=True):
    idxs = np.array(idxs.to('cpu'))
    assert img_shape[0] == idxs.shape[0] and img_shape[1] == idxs.shape[1]
                          ##road         ##sidelwalk    building        fence       p-railing       pole      traffic-light
    colors = np.array([[162, 75, 175], [244,31,181], [113,103,112], [100,75,44], [147,98,39], [203,202,190], [219,169,42],
                 [255,246,69], [21,149,18], [155,243,154], [39,201,200], [234,22,47], [255,0,30], [39,62,163]])
                   #traffic-sign    ##tree       vegetation       sky         person       rider         car
    colors = np.append(colors, ([[78,105,221], [128,147,224], [82,96,159], [94,19,36], [123,48,66], [0,0,0]]))
                                   #truck           bus           train       motorcycle   bicycle     misc
    colors = colors.reshape((20,3))
    
    
    cmap = torch.zeros((3,img_shape[0], img_shape[1])).long()
    
    for i in range(len(colors)):
        cmap[0,idxs==i] = colors[i][0]
        cmap[1,idxs==i] = colors[i][1]
        cmap[2,idxs==i] = colors[i][2]
    
    cmap = cmap.permute(1,2,0)
    if plot:
        fig = plt.figure(figsize=(figsize[0], figsize[1]))
        ax = fig.add_subplot()
        ax.imshow(cmap)
    
    return cmap

def make_color_pred(model, Xi, img_shape, figsize=(20,20), plot=True):
    idxs = make_pred(model, Xi, img_shape)
    return colorify(idxs, img_shape, figsize=figsize, plot=plot)