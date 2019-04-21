import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import ConvLSTM
from utils import (GetData, GetDataLoader, GetNext, NormalizeImage,
                   VectorizeDataset)

BATCH_SIZE = 8
ITERATIONS = 10000
LEARNING_RATE = 1e-2


def Train(model, x, y, optimizer):
    model.cuda()
    model.train()

    predictions = model(x)

    y = th.from_numpy(y.astype(np.float32)).cuda()
    loss = F.binary_cross_entropy_with_logits(predictions, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.item()


def Test(model, x, y):
    model.cuda()
    model.eval()

    predictions = model(x)

    y = th.from_numpy(y.astype(np.float32)).cuda()
    loss = F.binary_cross_entropy_with_logits(predictions, y)

    predictions = predictions >= 0.5
    predictions = predictions.detach().cpu().numpy()

    return loss.item(), predictions


if(__name__ == "__main__"):
    with open("results.txt", "w") as file:
        dataset_x, dataset_y = GetData("Data")
        index_dataset, hashmap_x, hashmap_y = VectorizeDataset(
            dataset_x, dataset_y, 9)

        train_indices, test_indices = train_test_split(
            index_dataset, test_size=0.3)

        train_mini_batches = train_indices.shape[0] // BATCH_SIZE
        test_mini_batches = test_indices.shape[0] // BATCH_SIZE

        train_dataloader = GetDataLoader(
            hashmap_x, hashmap_y, train_indices, BATCH_SIZE)
        test_loader = GetDataLoader(hashmap_x, hashmap_y, test_indices, BATCH_SIZE)

        del dataset_x, dataset_y, hashmap_x, hashmap_y, index_dataset

        model = ConvLSTM(
            input_channels=4,
            hidden_channels=[4, 8, 16],
            kernel_size=5
        )

        optimizer = optim.Adamax(model.parameters(), lr=LEARNING_RATE)

        for iteration in range(ITERATIONS):
            mini_batch_x, mini_batch_y = GetNext(train_dataloader)

            if(iteration % 100 == 0):
                loss = 0
                count = 0
                model_predictions = []
                truths = []
                test = True
                for count in tqdm(range(test_mini_batches)):
                    mini_batch_x, mini_batch_y = GetNext(test_loader)
                    loss_, predictions = Test(model, mini_batch_x, mini_batch_y)
                    model_predictions.append(predictions)
                    truths.append(mini_batch_y)
                    loss += loss_
                
                count += 1
                model_predictions = np.concatenate(model_predictions, axis=0)
                model_predictions = model_predictions.reshape(-1)
                truths = np.concatenate(truths, axis=0)
                truths = truths.reshape(-1)

                accuracy = accuracy_score(truths, model_predictions)
                precision = precision_score(truths, model_predictions)
                recall = recall_score(truths, model_predictions)
                conf_mat = confusion_matrix(truths, model_predictions)
                
                print("Test Loss =", loss / count)
                print("Test accuracy =", accuracy)
                print("Test precision =", precision)
                print("Test recall =", recall)
                print("Test confusion =")
                print(conf_mat)
                print()
                
                file.write("Test Loss = " + str(loss / count) + "\n")
                file.write("Test accuracy = " + str(accuracy) + "\n")
                file.write("Test precision = " + str(precision) + "\n")
                file.write("Test recall = " + str(recall) + "\n")
                file.write("Test confusion =\n")
                file.write(str(conf_mat) + "\n\n")
            
            loss = Train(model, mini_batch_x, mini_batch_y, optimizer)
            print("Train minibatch loss = {:.17f} @ iteration".format(
                loss), iteration)
