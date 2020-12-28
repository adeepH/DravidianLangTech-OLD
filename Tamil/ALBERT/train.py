# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1-Kutr_DWS9fi2oHtwsiTtIHNDO73wrMh
"""
#importing the necessary libraries
from utils import *
from dataset import SharedTask
from model import SharedTaskClassifier
import pandas as pd
import torch
import torch.nn as nn
from transformers import AdamW,get_linear_schedule_with_warmup,AutoModel,AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
train = pd.read_csv('/home/adweeb/pytorch/shared_task/English_train.csv',#delimiter=',',
                 header=None,names=['id','text','label'])
train = train.drop(columns='id')
train= train[1:]
train.label = train.label.apply({'fake':0,'real':1}.get)
train.head(10)

val = pd.read_csv('/home/adweeb/pytorch/shared_task/English_val.csv',
                  header=None,names=['id','text','label'])
val = val.drop(columns='id')
val = val[1:]
val.label = val.label.apply({'fake':0,'real':1}.get)
val.head(9)

print('Training set size:',train.shape)
#Uncomment the next line when we have the test data
#print('Testing set size:',test.shape)
print('validation set size:',val.shape)

#initialize the parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
PRE_TRAINED_MODEL_NAME = 'albert-base-v2'
tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

BATCH_SIZE = 32
MAX_LEN = 128
train_data_loader = create_data_loader(train,tokenizer,MAX_LEN,BATCH_SIZE)
val_data_loader = create_data_loader(val,tokenizer,MAX_LEN,BATCH_SIZE)

albert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

model = SharedTaskClassifier(2)
model = model.to(device)

EPOCHS = 2 #increase the epoch time to 10 for better results
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

from collections import defaultdict
import torch
 
history = defaultdict(list)
best_accuracy = 0
for epoch in range(EPOCHS):
 
 
  start_time = time.time()
  train_acc,train_loss = train_epoch(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler,
      len(train)
  )
   
  end_time = time.time()
  epoch_mins, epoch_secs = epoch_time(start_time, end_time)
  print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
  print(f'Train Loss {train_loss} accuracy {train_acc}')
  print()

  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)

""" Use it to save the state of the model and load the model
  if val_acc > best_accuracy:
    torch.save(model.state_dict(),'albert-base-v2.bin')
    best_accuracy = val_acc
"""
val_acc, val_loss = eval_model(
  model,
  val_data_loader,
  loss_fn,
  device,
  len(val) #Change it to test when you have the test results
)
print(f'The validation accuracy: {val_acc} | validation Loss : {val_loss}')
y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  val_data_loader,
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
)

class_name = ['fake','real'] #change accordingly for the Hindi task

print(classification_report(y_test, y_pred, target_names=class_name,zero_division=0))

#prints the confusion matrix
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_name, columns=class_name)
show_confusion_matrix(df_cm)

