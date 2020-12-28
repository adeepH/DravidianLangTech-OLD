import pandas as pd
from torch.utils.data import Dataset,DataLoader
import time
import seaborn as sns
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from dataset import SharedTask
import matplotlib.pyplot as plt
#from dataset import SharedTask
def create_data_loader(df,tokenizer,max_len,batch_size):
	ds = SharedTask(
		text = df.text.to_numpy(),
		label = df.label.to_numpy(),
		tokenizer = tokenizer,
		max_len = max_len
		)

	return DataLoader(ds,
		batch_size = batch_size,
		shuffle = True,
		num_workers=4)

def train_epoch(model,data_loader,loss_fn,optimizer,device,scheduler,n_examples):
	model = model.train()
	losses = []
	correct_predictions = 0

	for data in data_loader:
		input_ids = data['input_ids'].to(device)
		attention_mask = data['attention_mask'].to(device)
		labels = data['label'].to(device)

		outputs = model(
		    input_ids=input_ids,
		    attention_mask=attention_mask
		    )
		_, preds = torch.max(outputs, dim=1)
		loss = loss_fn(outputs,labels)

		correct_predictions += torch.sum(preds == labels)
		losses.append(loss.item())

		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		scheduler.step()
		optimizer.zero_grad()

	return correct_predictions.double() / n_examples, np.mean(losses)
	
def eval_model(model, data_loader, loss_fn, device, n_examples):

	  model = model.eval()
	  losses = []
	  correct_predictions = 0
	  with torch.no_grad():
	    for d in data_loader:
	      input_ids = d["input_ids"].to(device)
	      attention_mask = d["attention_mask"].to(device)
	      labels = d["label"].to(device)
	      outputs = model(
		input_ids=input_ids,
		attention_mask=attention_mask
	      )
	      _, preds = torch.max(outputs, dim=1)
	      loss = loss_fn(outputs, labels)
	      correct_predictions += torch.sum(preds == labels)
	      losses.append(loss.item())
	  return correct_predictions.double() / n_examples, np.mean(losses)
	
def epoch_time(start_time,end_time):
	elapsed_time = end_time - start_time
	elapsed_mins = int(elapsed_time/60)
	elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
	return elapsed_mins,elapsed_secs

def get_predictions(model,data_loader,device):

	model = model.eval()
	sentence = []
	predictions = []
	prediction_probs = []
	real_values = []
	
	with torch.no_grad():
		for d in data_loader:
			text = d['text']
			input_ids = d['input_ids'].to(device)
			attention_mask = d['attention_mask'].to(device)
			labels = d['label'].to(device)
			
			outputs = model(
				input_ids = input_ids,
				attention_mask = attention_mask
				)
			
			_,preds = torch.max(outputs,dim=1)
			sentence.extend(text)
			predictions.extend(preds)
			prediction_probs.extend(outputs)
			real_values.extend(labels)
			
	predictions = torch.stack(predictions).cpu()
	prediction_probs = torch.stack(prediction_probs).cpu()
	real_values = torch.stack(real_values).cpu()
	
	return sentence,predictions,prediction_probs,real_values

def predict_label(sentence,MAX_LEN,tokenizer,model,class_name,device):
	
	encoded_sentence = tokenizer.encode_plus(
		sentence,
		max_length = MAX_LEN,
		add_special_tokens = True,
		return_token_type_ids = False,
		padding = 'max_length',
		return_attention_masks = True,
		return_tensors = 'pt',
		truncation = True
		)
	input_ids = encoded_sentence['input_ids'].to(device)
	attention_mask = encoded_sentence['attention_mask'].to(device)
	output = model(input_ids,attention_mask)
	_,predictions = torch.max(output,dim=1)
	return sentence,class_name['prediction']
	 
def show_confusion_matrix(confusion_matrix):
	hmap = sns.heatmap(confusion_matrix,annot= True, fmt="d",cmap="Blues")
	hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
	hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=0, ha='right')
	plt.ylabel('True label')
	plt.xlabel('predicted label')
			
