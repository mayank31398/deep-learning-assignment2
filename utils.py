import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

plt.switch_backend("QT5Agg")

def Plot():
    def ImageDerivative(image):
        slide = np.zeros((240, 320, 4), dtype=np.uint8)
        slide[..., :3] = image
        slide[..., 3] = cv2.Canny(image, 100, 100)

        return slide
    
    frame = cv2.imread("Data/frames/lec1/l1s00027.jpg")
    frame_ = cv2.imread("Data/frames/lec1/l1s00026.jpg")

    frame = frame.astype(np.float32) - frame_.astype(np.float32)
    frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame))
    
    fig = plt.figure()
    plt.imshow(frame)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.subplots_adjust(
        top=0.98,
        bottom=0.05,
        left=0.05,
        right=0.992,
        hspace=0.2,
        wspace=0.2
    )
    
    plt.savefig("frame_.png")


def GetData(path):
    def ImageDerivative(image):
        slide = np.zeros((240, 320, 4), dtype=np.uint8)
        slide[..., :3] = image
        slide[..., 3] = cv2.Canny(image, 100, 100)

        return slide

    lectures = os.listdir(os.path.join(path, "frames"))
    dataset_x = []
    dataset_y = []

    for lecture in tqdm(lectures[:15], desc="Loading data"):
        frames = os.listdir(os.path.join(path, "frames", lecture))
        data_x = []

        for frame in frames:
            frame = cv2.imread(os.path.join(path, "frames", lecture, frame))
            frame = cv2.resize(frame, (320, 240))
            frame = ImageDerivative(frame)
            frame = frame.reshape(-1)

            frame = frame.astype(np.float32) / 255
            frame = frame.astype(np.float32)

            data_x.append(frame)

        dataset_x.append(data_x)

        data_y = pd.read_csv(os.path.join(
            path, "labels", lecture + ".csv"), header=None)
        dataset_y.append(data_y[0].tolist())

    return dataset_x, dataset_y


def VectorizeDataset(dataset_x, dataset_y, context):
    def GetHashmap(dataset_x, dataset_y):
        hashmap_x = {}
        hashmap_y = {}

        index = 1
        for i in range(len(dataset_x)):
            lecture_x = dataset_x[i]
            lecture_y = dataset_y[i]

            for j in range(len(lecture_x)):
                image = lecture_x[j]
                label = lecture_y[j]

                hashmap_x[index] = image
                hashmap_y[index] = label
                index += 1

        hashmap_x[0] = np.zeros(hashmap_x[1].shape)
        hashmap_y[0] = 0

        hashmap_x = pd.DataFrame(hashmap_x)
        hashmap_y = pd.Series(hashmap_y)

        return hashmap_x, hashmap_y
    
    hashmap_x, hashmap_y = GetHashmap(dataset_x, dataset_y)

    index = 1
    indices = []
    for i in range(len(dataset_x)):
        indices_ = []
        for j in range(len(dataset_x[i])):
            indices_.append(index)
            index += 1
        indices_ = context * [0] + indices_
        indices.append(indices_)

    index_dataset = []
    for i in range(len(indices)):
        for j in range(0, len(indices[i]) - context - 1, context):
            index_dataset.append(indices[i][j: j + context + 1])
    index_dataset = np.array(index_dataset)

    # index_dataset = []
    # for i in range(len(indices)):
    #     for j in range(len(indices[i]) - context - 1):
    #         index_dataset.append(indices[i][j: j + context + 1])
    # index_dataset = np.array(index_dataset)

    return index_dataset, hashmap_x, hashmap_y


def GetDataLoader(hashmap_x, hashmap_y, index_dataset, batch_size):
    def GetBatch(hashmap_x, hashmap_y, mini_batch_indices):
        old_shape = mini_batch_indices.shape
        mini_batch_indices = mini_batch_indices.reshape(-1)

        x_ = hashmap_x[mini_batch_indices].values
        x_ = x_.reshape(240, 320, 4, old_shape[0], old_shape[1])
        x_ = x_.transpose(3, 4, 0, 1, 2)

        y_ = hashmap_y[mini_batch_indices].values
        y_ = y_.reshape(old_shape)

        return x_, y_

    while(True):
        np.random.shuffle(index_dataset)
        num_batches = index_dataset.shape[0] // batch_size
        for i in range(num_batches):
            mini_batch_indices = index_dataset[i *
                                               batch_size: (i + 1) * batch_size]
            x_, y_ = GetBatch(hashmap_x, hashmap_y, mini_batch_indices)
            yield x_.astype(np.float32), y_.astype(np.float32)


def GetNext(dataloader):
    mini_batch_x, mini_batch_y = next(dataloader)
    mini_batch_x = mini_batch_x.transpose(0, 1, 4, 2, 3)
    mini_batch_x = NormalizeImage(mini_batch_x)
    return mini_batch_x, mini_batch_y


def NormalizeImage(x):
    x = 2 * x - 1
    return x


if(__name__ == "__main__"):
    Plot()