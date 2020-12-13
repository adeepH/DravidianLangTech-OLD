import torch.nn as nn
from transformers import AutoModel
pretrained_model = 'albert-base-v2'
class SharedTaskClassifier(nn.Module):
	
	def __init__(self,n_classes):
		super(SharedTaskClassifier,self).__init__()
		self.auto = AutoModel.from_pretrained('albert-base-v2')
		self.drop = nn.Dropout(p=0.4)
		self.out1 = nn.Linear(self.auto.config.hidden_size,128)
		self.drop1 = nn.Dropout(p=0.4)
		self.relu = nn.ReLU()
		self.out = nn.Linear(128,n_classes)
		
	def forward(self, input_ids, attention_mask):
		
		_, pooled_output = self.auto(
			input_ids = input_ids,
			attention_mask = attention_mask
		)
		
		output = self.drop(pooled_output)
		output = self.out1(output)
		output = self.relu(output)
		output = self.drop1(output)
		return self.out(output)
