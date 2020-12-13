import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset,DataLoader

class SharedTask(Dataset):


#""" custom dataset should inherit Dataset and override the following methods:
#    __len__ so that len(dataset) returns the size of the dataset.
#    __getitem__ to support the indexing such that dataset[i] can be used to get ith sample
#"""
# The convention in BERT is:
   # (a) For sequence pairs:
   #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
   #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
   # (b) For single sequences:
   #  tokens:   [CLS] the dog is hairy . [SEP]
   #  type_ids:   0   0   0   0  0     0   0
   #
   # Where "type_ids" are used to indicate whether this is the first
   # sequence or the second sequence. The embedding vectors for `type=0` and
   # `type=1` were learned during pre-training and are added to the wordpiece
   # embedding vector (and position vector). This is not *strictly* necessary
   # since the [SEP] token unambiguously separates the sequences, but it makes
   # it easier for the model to learn the concept of sequences.
   #
   # For classification tasks, the first vector (corresponding to [CLS]) is
   # used as as the "sentence vector".


	def __init__(self,sentence,label,tokenizer,max_len):
		self.sentence = sentence
		self.label = label
		self.tokenizer = tokenizer
		self.max_len = max_len
	
	def __len__(self):
		return len(self.sentence)	
	
	def __getitem__(self,item):
		sentence = str(self.sentence[item])
		label = self.label[item]
		
		encoding = self.tokenizer.encode_plus(
			sentence,
			add_special_tokens = True,#to encode the sequences with the special tokens relative to their model.
			max_length = self.max_len,#set the max_len to the max input of BERT(512)
			return_token_type_ids = False,
			padding ='max_length' , #Inputs are padded to max_length of 512 if len(sentence<512)
			return_attention_mask = True,
			return_tensors = 'pt',
			truncation = True
		)
		
		
		return {
			'sentences' : sentence, # returns the input_ids
			'input_ids' : encoding['input_ids'].flatten(), #returns the attention masks
			'attention_mask' : encoding['attention_mask'].flatten(),
			'label' : torch.tensor(label,dtype=torch.long)
			
		}
