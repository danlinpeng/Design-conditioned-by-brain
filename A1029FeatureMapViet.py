import numpy as np
import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.backends.cudnn as cudnn; cudnn.benchmark = True
import load_data as dp
import pickle

# Define parameters
parser = argparse.ArgumentParser(description = "Classifier")


# Model select 
parser.add_argument('--model_selection', default='lstm', help='choose the classifier model')
parser.add_argument('--continue_train', default=False, help='need to continue train the model')
parser.add_argument('--model_path', help='the path of the model')
# Define LSTM
parser.add_argument('--n_lstm', default=1, type=int,help='number of lstm')
parser.add_argument('--lstm_layer', default=1, type=int, help='number of lstm hidden layers')
parser.add_argument('--lstm_input', default=64, type=int, help='lstm hidden layer size')
parser.add_argument('--lstm_output', default=64, type=int, help='output size')
parser
# Define CNN
parser.add_argument('--1DConv', default=2, type=int, help ='number of 1D conv layers')
parser.add_argument('--2DConv', default=2, type=int, help ='number of 2D conv layers')

parser.add_argument('--n_class', default=100, type=int, help = 'number of classes')

# Training options
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--optimizer', default='Adam', help='Adam|SGD')
parser.add_argument('--learning_rate',default=0.0001, type=float)
parser.add_argument('--learning_rate_decay', default=0.5, type=float, help='lr decay factor')
parser.add_argument('--learning_rate_every', default=100, type=int, help='lr decay per')
parser.add_argument('--epochs', default=3000, type=int, help='number of epochs')

# Dataset
parser.add_argument('--brain_signal_file', default='datasets/')
# Other
parser.add_argument('--is_train',default='train',help='train|generate')
# Cuda
cuda = True if torch.cuda.is_available() else False




opt = parser.parse_args()
print(opt)

class LSTM_Model(nn.Module):
    
    def __init__(self, num_layers, input_size, hidden_size, output_size,n_class,n_lstm):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_class = n_class
        self.n_lstm = n_lstm
        # Define layers
        self.lstm = nn.LSTM(hidden_size, output_size, num_layers=num_layers, batch_first=True)
        self.layer = nn.Sequential(nn.Linear(64,256),
                                   nn.LeakyReLU(),
                                   nn.Linear(256,n_class),
                                   )

    def forward(self, x):
        # initial 
        batch_size = x.size(0)
        lstm_init = [torch.zeros(self.num_layers,batch_size, self.output_size),torch.zeros(self.num_layers,batch_size, self.output_size)]
        if cuda :
            lstm_init = (lstm_init[0].cuda(),lstm_init[1].cuda())
        lstm_init = (Variable(lstm_init[0], requires_grad=x.requires_grad), Variable(lstm_init[1], requires_grad=x.requires_grad))
        
        # Forword LSTM
        for i in range(self.n_lstm):
            x = self.lstm(x, lstm_init)
        x = x[0][:,-1,:]
        x = self.layer(x)
        return x

# Prepare model
if opt.model_selection == 'lstm':
    model = LSTM_Model(opt.lstm_layer, opt.lstm_input, opt.lstm_input, opt.lstm_output, opt.n_class, opt.n_lstm)

optimizer = getattr(torch.optim, opt.optimizer)(model.parameters(), lr = opt.learning_rate)

mse = nn.MSELoss()
if opt.is_train == 'generate':
    model = torch.load("model_backup.pth")
    
if cuda: model.cuda()

# Prepare training data  
train=dp.load_data(opt.batch_size) 

if opt.is_train == 'generate':
    result=dict()
    for s in ("train","valid","test"):
            data=train[s]
            result[s]={'target_feature':[],'feature':[],'label':[],'img_name':[]}
                
             # load data
            for index,batch_data in enumerate(data):
                input = torch.from_numpy(batch_data['data'])

                # Wrap for autograd
                input = Variable(input, requires_grad  = (s != "train")).float()
                # Forward
                output = model(input)
                if index==0:
                    result[s]['feature']=output.data.numpy()
                    result[s]['label']=batch_data['label']
                    result[s]['target_feature']=batch_data['target_feature']
                    result[s]['img_num']=batch_data['img_num']
                else:
                    result[s]['feature']=np.vstack((result[s]['feature'],output.data.numpy()))
                    result[s]['label']=np.concatenate([result[s]['label'],batch_data['label']])
                    result[s]['target_feature'] = np.vstack((result[s]['target_feature'], batch_data['target_feature']))
                    result[s]['img_num']=np.concatenate([result[s]['img_num'],batch_data['img_num']])
   #save file
    with open ('features.pkl','wb') as f:
        pickle.dump(result,f)
else:
    # Start training
    for epoch in range(opt.epochs):
        # use dictionary to record the loss and accuracy
        ac = {'train':0, 'valid':0, 'test':0}
        losses = {'train':0, 'valid':0, 'test':0}
        count = {'train':0, 'valid':0, 'test':0}
        # Learning rate decay for SGD
        if opt.optimizer == "SGD":
            lr = opt.learning_rate * (opt.learning_rate_decay_by ** (epoch // opt.learning_rate_decay_every))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
     # training all batches
        for s in ("train","valid","test"):
            data=train[s]
            if s == 'train': 
                model.train()
            else:
                model.eval()
                
             # load data
            for _,batch_data in enumerate(data):
                input = torch.from_numpy(batch_data['data'])
                target = torch.from_numpy(batch_data['target_feature'])
                
                # Check CUDA
                if cuda:
                    input = input.cuda(async = True)
                    target = target.cuda(async = True)
                # Wrap for autograd
                input = Variable(input, requires_grad  = (s != "train")).float()
                target = Variable(target, requires_grad = (s != "train")).float()
                # Forward
                output = model(input)
                loss = mse(output, target)
                losses[s] += loss.item()
                count[s] += input.data.size(0)
    
                # Backward and optimize
                if s == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()       
    
        # Print info at the end of the epoch
        content = "Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}".format(epoch,losses["train"]/count["train"],ac["train"]/count["train"],losses["valid"]/count["valid"],ac["valid"]/count["valid"],losses["test"]/count["test"],ac["test"]/count["test"])
        
        print(content)
       
        torch.save(model,"model_backup.pth") 

 
