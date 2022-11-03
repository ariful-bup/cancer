#!/usr/bin/env python
# coding: utf-8

# In[39]:


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from utilities.dataloader import get_train_test_loaders, get_cv_train_test_loaders
from utilities.model import CustomVGG
from utilities.helper import train, evaluate, predict_localize
from utilities.constants import NEG_CLASS


# ## Parameters

# In[40]:


data_folder = "D:/Pythonenv/Defect Detection/Data/leather (1)/"
subset_name = "leather"
data_folder = os.path.join(data_folder, subset_name)

batch_size = 10
target_train_accuracy = 0.98
lr = 0.0001
epochs = 100
class_weight = [1, 3] if NEG_CLASS == 1 else [3, 1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

heatmap_thres = 0.7
n_cv_folds = 5


# # Data

# In[41]:


train_loader, test_loader = get_train_test_loaders(
    root=data_folder, batch_size=batch_size, test_size=0.2, random_state=42,
)


# In[42]:


print(train_loader)
print(test_loader)


# # Model Training

# In[43]:


model = CustomVGG()

class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.Adam(model.parameters(), lr=lr)


# In[ ]:


model.summary()


# In[44]:


model = train(
    train_loader, model, optimizer, criterion, epochs, device, target_train_accuracy
)


# In[45]:


#model_path = model.h5"
torch.save(model, 'D:\\Pythonenv\\Defect Detection\\Data\\leather (1)\\leather\\custom.h5')
# model = torch.load(model_path, map_location=device)


# # Evaluation

# In[46]:


evaluate(model, test_loader, device)


# # Cross Validation

# In[ ]:


cv_folds = get_cv_train_test_loaders(
    root=data_folder,
    batch_size=batch_size,
    n_folds=n_cv_folds,
)

class_weight = torch.tensor(class_weight).type(torch.FloatTensor).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weight)

for i, (train_loader, test_loader) in enumerate(cv_folds):
    print(f"Fold {i+1}/{n_cv_folds}")
    model = CustomVGG(2) #input_Size=2
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = train(train_loader, model, optimizer, criterion, epochs, device)
    evaluate(model, test_loader, device)


# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt


# # Visualization

# In[20]:


predict_localize(
    model, test_loader, device, thres=heatmap_thres, n_samples=15, show_heatmap=False
)

