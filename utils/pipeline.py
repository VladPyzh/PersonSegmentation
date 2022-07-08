import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from collections import defaultdict
from tqdm.notebook import tqdm
from IPython.display import clear_output

from .metrics import *


device = f"cuda" if torch.cuda.is_available() else "cpu"

def plot_learning_curves(history):
    '''
    Функция для вывода лосса и метрики во время обучения.
    :param history: (dict)
    accuracy и loss на обучении и валидации
    '''
    fig = plt.figure(figsize=(20, 7))

    plt.subplot(1,2,1)
    plt.title('loss', fontsize=15)
    plt.plot(history['loss']['train'], label='train')
    plt.plot(history['loss']['val'], label='val')
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('эпоха', fontsize=15)
    plt.legend()


    plt.subplot(1,2,2)
    plt.title('dice', fontsize=15)
    plt.plot(history['dice']['train'], label='train')
    plt.plot(history['dice']['val'], label='val')
    plt.ylabel('dice', fontsize=15)
    plt.xlabel('эпоха', fontsize=15)
    plt.legend()

    plt.show()


def train_with_aug(
    model,
    criterion,
    optimizer,
    scheduler,
    train_batch_gen,
    val_batch_gen,
    num_epochs=50,
):
    '''
    Функция для обучения модели и вывода лосса и метрики во время обучения.
    :param model: обучаемая модель
    :param criterion: функция потерь
    :param optimizer: метод оптимизации
    :param train_batch_gen: генератор батчей для обучения
    :param val_batch_gen: генератор батчей для валидации
    :param num_epochs: количество эпох
    :return: обученная модель
    :return: (dict) accuracy и loss на обучении и валидации ("история" обучения
    '''

    best_dice = None
    
    history = defaultdict(lambda: defaultdict(list))
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        train_dice = 0
        
        val_loss = 0
        val_dice = 0
        
        start_time = time.time()

        model.train(True)

        for X_batch, y_batch in tqdm(train_batch_gen):
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss += loss.detach().cpu().numpy()
            y_pred = logits.detach().cpu().numpy().argmax(axis=1)
            train_dice += get_dice(y_batch.cpu().numpy(), y_pred) 

        train_loss /= len(train_batch_gen)
        train_dice /= len(train_batch_gen)
        history['loss']['train'].append(train_loss)
        history['dice']['train'].append(train_dice)

        scheduler.step()
        
        model.train(False)

        for X_batch, y_batch in tqdm(val_batch_gen):
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            
            val_loss += loss.detach().cpu().numpy()
            y_pred = logits.detach().cpu().numpy().argmax(axis=1)
            val_dice += get_dice(y_batch.cpu().numpy(), y_pred)

        val_loss /= len(val_batch_gen)
        val_dice /= len(val_batch_gen)
        history['loss']['val'].append(val_loss)
        history['dice']['val'].append(val_dice)
        
        
        if best_dice is None:
          best_dice = val_dice
          torch.save(model.state_dict(),'/home/vapyzh/FaceSegmentationReport/weights/weights_'\
           + "_" + str(time.time()))
        elif best_dice < val_dice:
          best_dice = val_dice
          torch.save(model.state_dict(),'/home/vapyzh/FaceSegmentationReport/weights/weights_'\
           + "_" + str(time.time()))


        clear_output()
        
        #beatiful print time and metrics
        print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
        print(" training loss (in-iteration): \t{:.6f}".format(train_loss))
        print(" validation loss (in-iteration): \t{:.6f}".format(val_loss))
        print(" training dice: \t\t\t{:.2f} %".format(train_dice * 100))
        print(" validation dice: \t\t\t{:.2f} %".format(val_dice * 100))
        
        plot_learning_curves(history)

        X_batch, y_batch = next(iter(val_batch_gen))
        orig = X_batch[0]
        true = y_batch[0]
        logits = model(orig.unsqueeze(0).float().to(device))
        pred = logits.detach().cpu().numpy()
        predicted_mask = pred.argmax(axis=1)

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(orig.permute(1, 2, 0), label="orig pic")
        plt.subplot(1, 3, 2)
        plt.imshow(true.squeeze(), label="true mask")
        plt.subplot(1, 3, 3)
        plt.imshow(predicted_mask[0], label="predicted mask")
        plt.show()


    return model, history    