import os
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
# from torch_geometric.data import DataLoader # old
from torch_geometric.loader import DataLoader # updated
from GraphNetCellModel import myPNA
from GraphNetCellUtils import getBaseline, visualizeDataDistribution, visualizeDataGraph
import scipy.stats
import seaborn as sns
from scipy import io
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
print(torch.cuda.is_available())

'''
params
'''
RandSeedNumber = 12345

ablation_type = "Full" # This is the default
# ablation_type = "Positional"
# ablation_type = "Topological"
# ablation_type = "RemoveArea"
# ablation_type = "RemovePeri"
# ablation_type = "RemoveEdgeEntirely"

lagTime         = 3 # ref value: 3. Range: 0 to 6
test_idx_lst    = [16,17]# there are 20 positions in total. By detault, we use 16, 17 as the test set, and the rest as the train set

num_layers      = 5     # ref value: 5
hidden_channels = 36    # ref value: 36
batch_size      = 128   # ref value: 128
learning_rate   = 1e-5  # ref value: 1e-5
weight_decay    = 1e-3  # ref value: 1e-3
EpochNum        = 501   # ref value: 501


'''
Functions
'''
import random
# function for setting random seed number
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # GPU will be slower but deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # True
    print("Seeded everything: {}".format(seed))


# function for loading data
import pickle
def load_dataset(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        return data


# function for picking mean displacement at a given lag time as the output
def setOutput(data_list,lagTime):
    for data in data_list:
            data.y = data.meanDisp[0][lagTime]
    return data_list

# function for choosing the node and edge information as the input
def setInput(data_list, ablation_type):
    for data in data_list:
        
        # assign data.area and data.perimeter
        # this is only for visualization
        data.area = data.x[:,0]
        data.perimeter = data.x[:,1]

        # Assign input
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
            # replace edges by self loops
            temp = torch.arange(0, data.x.shape[0], dtype=torch.long).unsqueeze(0).repeat(2, 1)
            data.edge_index = temp
        elif ablation_type == "Full":
            pass
        else:
            raise ValueError("Invalid ablation type")
    return data_list



def train():
    model.train()
    for data in train_loader:
        optimizer.zero_grad()

        # calculate loss
        pred = model(data.x.cuda(), data.edge_index.cuda(), data.batch.cuda())
        loss = criterion(pred.squeeze(), data.y.squeeze().cuda())

        # update
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return pred.squeeze().detach().cpu(), data.y.squeeze()


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



'''
Main
'''
# create folders for saving results
os.makedirs('results', exist_ok=True)
os.makedirs('weights', exist_ok=True)

# rand seed initialization
set_seed(RandSeedNumber)

# load data
data_list_all = load_dataset('GNNcell_AllExpData.pkl')

# train/test spit
data_list_train = []
data_list_test = []
for i, item in enumerate(data_list_all):
    if i in test_idx_lst:
        data_list_test.extend(item)
    else:
        data_list_train.extend(item)

print(f'Train size:{len(data_list_train)}, Test size:{len(data_list_test)}')

# Select the mean displacement at one lag time scale as the output
data_list_train = setOutput(data_list_train,lagTime)
data_list_test  = setOutput(data_list_test,lagTime)

# Select the node features and edges as the input
data_list_train = setInput(data_list_train, ablation_type)
data_list_test  = setInput(data_list_test, ablation_type)

# Data loader
g = torch.Generator()
g.manual_seed(RandSeedNumber)
train_loader = DataLoader(data_list_train, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)
test_loader = DataLoader(data_list_test, batch_size=4096, shuffle=False, generator=g, drop_last=False) # Here we just use a large enough batch size to load all test data in a single batch

# Some initial visualization
# visualizeDataDistribution(data_list_train, data_list_test)
visualizeDataGraph(data_list_test, np.linspace(0, len(data_list_test)-1, num=5, endpoint=True, dtype=int))


# initialize model
in_channels = data_list_train[0].x.shape[1]
model = myPNA(data_list=data_list_train, in_channels=in_channels, hidden_channels=hidden_channels, num_layers=num_layers).cuda()
print(model)

# initialize optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
use_amp = True
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)  # old
scaler = torch.amp.GradScaler('cuda', enabled=use_amp) # updated

'''
Now we train the model
'''
# train model
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



'''
Post-processing (plots and save results)
'''

# baseline
baseline_corr, baseline_mse = getBaseline(data_list_test)
print(f'Baseline corr: {baseline_corr:.4f}, Baseline mse: {baseline_mse:.4f}')
visualizeDataDistribution(data_list_train, data_list_test)

# 2D probability density plot
plt.figure(figsize=(4, 3))
# ax = sns.kdeplot(x=np.array(test_truth), y=np.array(test_pred), cmap="rocket_r", shade=True, fill=True, levels=10, cbar=True, cut=12)
ax = sns.kdeplot(x=np.array(test_truth), y=np.array(test_pred), cmap="rocket_r", fill=True, levels=10, cbar=True, cut=12)
plt.scatter(np.array(test_truth),np.array(test_pred), s=10, c='lightblue', alpha=0.6, label = 'Data')

# reference line
xref = np.linspace(0.0, 2.0, num=100)
plt.plot(xref,xref, c = 'k', label = 'Reference',linewidth = 1, linestyle = 'dashed')

plt.xlim([0.0,1.2])
plt.ylim([0.0,1.2])
plt.xticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
plt.xlabel('Mobility, $M$',fontsize=12)
plt.ylabel('Predicted mobility, $M_{NN}$',fontsize=12)
plt.tight_layout()
# plt.show()
plt.savefig(f'results/Pred_vs_truth.png', dpi=300, bbox_inches='tight')


# print results
print("Last corr:", test_corr_lst[-1], "Last Loss:", test_loss_lst[-1])
print("Maximum corr:", np.max(test_corr_lst), "Index of Maximum corr:", np.argmax(test_corr_lst))
print("Minimum Loss:", np.min(test_loss_lst), "Index of Minimum Loss:", np.argmin(test_loss_lst))


# save results
savePath = f'results/GraphNetCell_ExpData_{ablation_type}_layers{num_layers}_channels{hidden_channels}_batchSize{batch_size}_lr{learning_rate}_Epochs{EpochNum}_WeightDecay{weight_decay}_lag{lagTime}_seed{RandSeedNumber}.mat'
io.savemat(savePath, {'train_loss': train_loss_lst,  'train_corr': train_corr_lst,
                      'test_loss' : test_loss_lst,   'test_corr': test_corr_lst,
                      'baseline_loss': baseline_mse, 'baseline_corr': baseline_corr})
print(savePath)

# save trained model weights
torch.save(model.state_dict(), 'weights/GraphNetCellExp_weights.pth')