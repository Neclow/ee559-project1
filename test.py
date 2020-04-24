import torch
from torch import nn
import matplotlib.pyplot as plt
import time
from models import *
from train import train
from utils import load_data, weight_initialization, train_visualization

seed = 14
# Generate data
torch.manual_seed(seed)
train_loader, test_loader = load_data(seed=seed)

best_model = SiameseCNN(mode='dropout')
best_alpha = 1
best_alpha_decay = 1

# Apply training mode and weight initialization
best_model.train()
best_model.apply(weight_initialization)

# Train
start = time.time()
tr_loss, tr_acc, te_acc = train(best_model, train_loader, test_loader, alpha=best_alpha, 
                                alpha_decay=best_alpha_decay, verbose=True, plotting=True)
print('Training ended. Training time: %.2f s' % (time.time()-start))

print()
print('Train accuracy: %.4f. Test accuracy: %.4f' % (tr_acc[-1], te_acc[-1]))

train_visualization(best_model, tr_loss, tr_acc, te_acc)