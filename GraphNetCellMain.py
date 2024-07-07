
"""
Arguments
"""
RandSeedNumber = 7 # 7, 42, 12345
num_layers = 5 # ref value: 5
hidden_channels = 36 # ref value: 36
batch_size = 128 # ref value: 128
learning_rate = 1e-5 # ref value: 1e-5
weight_decay = 1e-3 # ref value: 1e-3
EpochNum = 501

lagTime = 3 # ref value: 3. Range: 0 to 6
ablation_type = "Full" # the main input type in the paper, with area and perimeter as nodal features
# ablation_type = "Positional" # uncomment to use (x,y) coordinates
# ablation_type = "Topological" # uncomment to remove nodal features
# ablation_type = "RemoveArea" # uncomment to remove area
# ablation_type = "RemovePeri" # uncomment to remove perimeter
# ablation_type = "RemoveEdgeEntirely" # uncomment to remove graph edges




"""
import package and set random seed
"""
import os
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import DataLoader
import seaborn as sns
import scipy.stats
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
print(os.getcwd())
!nvcc --version
print(torch.cuda.is_available())

import random
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # making sure GPU runs are deterministic even if they are slower
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True
    print("Seeded everything: {}".format(seed))

set_seed(RandSeedNumber)



"""
load data
"""
import pickle
def load_dataset(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        # return data[::5]# Load 1 in every 5 items from the dataset
        return data

data_list_train = load_dataset('TrainExpData20240619.pkl')
data_list_test = load_dataset('TestExpData20240619.pkl')

# use meanDisp at a given time scale as output y
def setOutput(data_list):
    for data in data_list:
            data.y = data.meanDisp[0][lagTime]
    return data_list

data_list_train = setOutput(data_list_train)
data_list_test  = setOutput(data_list_test)

# set input nodal features
def setInput(data_list):
    for data in data_list:
        # for visualization only
        data.area = data.x[:,0]
        data.perimeter = data.x[:,1]

        # for input
        if ablation_type == "Positional":
            temp = data.cell_pos
            data.x = temp
        elif ablation_type == "Topological":
            temp = torch.zeros_like(data.x)
            data.x = temp
        elif ablation_type == "RemoveArea":
            temp = data.x[:,1]
            temp = temp.unsqueeze(1)
            data.x = temp
        elif ablation_type == "RemovePeri":
            temp = data.x[:,0]
            temp = temp.unsqueeze(1)
            data.x = temp
        elif ablation_type == "RemoveEdgeEntirely":
            temp = torch.arange(0, data.x.shape[0], dtype=torch.long).unsqueeze(0).repeat(2, 1) # replace edges by self looping edge
            data.edge_index = temp
        elif ablation_type == "Full":
            pass
        else:
            raise ValueError("Invalid ablation type")
    return data_list

data_list_train = setInput(data_list_train)
data_list_test  = setInput(data_list_test)


# data loader
g = torch.Generator()
g.manual_seed(RandSeedNumber)

train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
test_loader = DataLoader(data_list_test, batch_size=batch_size, shuffle=False, generator=g, drop_last=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

for step, data in enumerate(test_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()



"""
Define model and train/test functions
"""

# define model
from GraphNetCellModel import myPNA
in_channels = data_list_train[0].x.shape[1]
model = myPNA(data_list=data_list_train, in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers).cuda()
print(model)

# train function
def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        pred = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())
        loss = criterion(pred.squeeze(), data.y.squeeze().cuda())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return pred.squeeze().detach().cpu(), data.y.squeeze()

# test function
@torch.no_grad()
def test(loader):
    model.eval()
    loss_list=[]
    pred_list=[]
    y_list=[]
    for data in loader:
        pred = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())
        loss = criterion(pred.squeeze().detach().cpu(), data.y.squeeze())
        loss_list.append(loss.item())
        pred_list.extend(pred.squeeze().detach().cpu())
        y_list.extend(data.y.squeeze())
    return loss_list, pred_list, y_list



"""
Train the model
"""
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

import warnings
warnings.filterwarnings('ignore')

train_loss_lst=[]
test_loss_lst=[]
train_corr_lst=[]
test_corr_lst=[]

for epoch in range(1, EpochNum):
    pred, truth = train()

    with torch.no_grad():
        train_loss, train_pred, train_truth = test(train_loader)
        test_loss, test_pred, test_truth = test(test_loader)
        train_loss_lst.append(sum(train_loss)/len(train_loss))
        test_loss_lst.append(sum(test_loss)/len(test_loss))

        # calculate correlation
        train_corr, train_p_value = scipy.stats.pearsonr(np.array(train_pred),np.array(train_truth))
        test_corr, test_p_value = scipy.stats.pearsonr(np.array(test_pred),np.array(test_truth))

        train_corr_lst.append(train_corr)
        test_corr_lst.append(test_corr)

        print(f'Epoch: {epoch:03d}, Train loss: {sum(train_loss)/len(train_loss):.4f}, Test loss: {sum(test_loss)/len(test_loss):.4f},\
            Pearson correlation (train): {train_corr:.4f}, Pearson correlation (test): {test_corr:.4f}')

    if epoch%50==0:
        plt.figure(figsize=(11, 3))

        plt.subplot(141)
        plt.plot(train_loss_lst, color='C1', label='train loss')
        plt.plot(test_loss_lst, color='C2', label='val loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')

        plt.subplot(142)
        plt.plot(train_corr_lst, color='C1', label='train corr')
        plt.plot(test_corr_lst, color='C2', label='val corr')
        plt.ylim([0,1])
        plt.xlabel('Epochs')
        plt.ylabel('Pearson correlation, test')
        plt.legend()

        plt.subplot(143)
        plt.scatter(np.array(train_truth),np.array(train_pred), s=5, c='k', alpha=0.5, label = 'data')
        xref = np.linspace(np.min(np.array(train_truth)), np.max(np.array(train_truth)), num=100)
        plt.plot(xref,xref, c = 'r', label = 'reference')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.legend()
        plt.title('Train')

        plt.subplot(144)
        plt.scatter(np.array(test_truth),np.array(test_pred), s=5, c='k', alpha=0.5, label = 'data')
        xref = np.linspace(np.min(np.array(test_truth)), np.max(np.array(test_truth)), num=100)
        plt.plot(xref,xref, c = 'r', label = 'reference')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.legend()
        plt.title('Validation')
        plt.tight_layout()
        plt.show()

        # probability density plot
        plt.figure(figsize=(4, 3))
        ax = sns.kdeplot(x=np.array(test_truth), y=np.array(test_pred), cmap="rocket_r", shade=True, fill=True, thresh=0, levels=10, cbar=True, cut=12)
        plt.scatter(np.array(test_truth),np.array(test_pred), s=10, c='lightblue', alpha=0.6, label = 'Data')

        xref = np.linspace(0.0, 2.0, num=100)
        plt.plot(xref,xref, c = 'k', label = 'Reference',linewidth = 1, linestyle = 'dashed')

        plt.xlim([0.1,0.9])
        plt.ylim([0.1,0.9])
        plt.xticks([0.1,0.3,0.5,0.7,0.9])
        plt.yticks([0.1,0.3,0.5,0.7,0.9])
        plt.xlabel('Mobility, $M$',fontsize=12)
        plt.ylabel('Predicted mobility, $M_{NN}$',fontsize=12)

        plt.show()

