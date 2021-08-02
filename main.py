from load_dataset import load_data
from split_data import split_data
from DataSetLoader import CustomDataset
from torch.utils.data import DataLoader
from model import BERTClass
from train_val import train,validation
from transformers import AutoTokenizer
import torch
from torch import cuda
import numpy as np
from sklearn import metrics


device = 'cuda' if cuda.is_available() else 'cpu'


data = load_data('data.csv')

n_classes = len(data.iloc[1,1])


model_name = "dccuchile/bert-base-spanish-wwm-cased"


MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-05

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset,test_dataset = split_data(data,0.8)

training_set = CustomDataset(train_dataset['text'],train_dataset['one_hot'], tokenizer, MAX_LEN)

testing_set = CustomDataset(test_dataset['text'],test_dataset['one_hot'], tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = BERTClass(n_classes,model_name)
model.to(device)

def loss_fn(outputs, targets):

    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):

    pwtrain(epoch,model,training_loader,device,optimizer,loss_fn)


for epoch in range(EPOCHS):

    outputs, targets = validation(epoch,model,testing_loader,device,optimizer,loss_fn)

    outputs = np.array(outputs) >= 0.5

    accuracy = metrics.accuracy_score(targets, outputs)
    
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    
    print(f"Accuracy Score = {accuracy}")
    
    print(f"F1 Score (Micro) = {f1_score_micro}")
    
    print(f"F1 Score (Macro) = {f1_score_macro}")
