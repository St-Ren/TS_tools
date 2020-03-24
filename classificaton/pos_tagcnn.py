




import numpy as np 

from torch.utils import data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import json
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn import metrics
#dataset loader 
class MDS(Dataset):
	def __init__(self,file_path='train',dict_path='tag_dic.txt',levels = list(range(5)),label = True):
		


		tag_dict = json.load(open(dict_path))
		self.ft_len = len(tag_dict)
		fts = []
		lb = []
		for i in levels:
			f= open(file_path+str(i)+'.tag')
			for line in json.load(f):
				ft = np.zeros([120,self.ft_len])
				for j in range(min(len(line),120)):
					if line[j] in tag_dict:
						#print(line[i],tag_dict[line[i]])
						ft[j][tag_dict[line[j]]] = 1
					else:
						for k in range(min(100,self.ft_len)):
							ft[j][k] = 1
				fts.append(ft)
				if label:
					lb.append(i)
		self.fts = np.array(fts)
		self.lb = np.array(lb)
	def dict_len(self):
		return self.ft_len
	def __len__(self):
		return len(self.lb)
	def __getitem__(self,idx):
		ft = self.fts[idx]
		label = self.lb[idx]

		return torch.FloatTensor(ft).unsqueeze(0), torch.tensor(label,dtype = torch.long)

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)

class Net(nn.Module):
    def __init__(self,dict_len):
        super(Net, self).__init__()
        self.dict_len = dict_len
        def conv(inp, oup,lenth = 1,bias = False):
            return nn.Sequential(
                nn.Conv2d(inp, oup, (3,lenth),padding = 0, bias=bias),
                nn.BatchNorm2d(oup),
                nn.LeakyReLU(inplace=True)

            )
        self.pool = nn.MaxPool2d((2,1))
       

        self.model = nn.Sequential(
            conv(1,32,self.dict_len),
            conv(32,32),
            self.pool,
            conv(32,64),
            conv(64,64),
            self.pool,
            conv(64,128),
            conv(128,128),
            self.pool,
            conv(128,256),
            conv(256,256)



        )
        self.fc = nn.Sequential(
        	nn.Linear(1792,256),
        	nn.Dropout(0.5),
        	nn.LeakyReLU(),
        	nn.Linear(256,32),
        	nn.Dropout(0.5),
        	nn.LeakyReLU(),
        	nn.Linear(32,5),        
        )
        self.output = nn.Softmax(dim=1)
        self.model.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)


    def forward(self, x):

        x = self.model(x)

        x = x.reshape(x.size(0),-1)
        x = self.fc(x)
        x = self.output(x)
        return x

       
      





def train(lr,epoch=1,batch=50,rep = 1,save_path = 'checkpoints/',data_path = './',save_interval = 1,model_path = None ):
	
	trdts = MDS(data_path+'train')
	valdts = MDS(data_path+'valid')
	best_loss = 10000000000
		
		
	
	net = Net(dict_len = 49)
	net = net.cuda()

	trainloader = DataLoader(trdts, batch_size=batch, shuffle=True, num_workers=4)
	valloader = DataLoader(valdts,batch_size = 300, num_workers = 4)
	#valloader = DataLoader(valdts,len(validc))
	loss_func = nn.CrossEntropyLoss()
	
	#opt = torch.optim.SGD(net.parameters(), lr=lr,momentum=0.9)
	opt = torch.optim.Adam(net.parameters(), lr=lr)
	if model_path != None:
		net.load_state_dict(torch.load(model_path))
		net.eval()
		opt.load_state_dict(torch.load(model_path+'.lr'))
		opt.eval()

	for ep in range(epoch):
		print(ep)
		ac= 0.0
		c = 0.0
		for batch_id, batch in enumerate(trainloader):
			#print(batch_id)
			fts, labels = batch
			#fts = tran(fts)
			fts = fts.cuda()
			labels = labels.cuda()
			#print('output')
			
			
			output = net(fts)
			#print(output)
			z = loss_func(output,labels)
			opt.zero_grad()
			z.backward()
			opt.step()
			#print('opt')
       
			c+=torch.sum(torch.max(output, 1)[1].cuda().data == labels).type(torch.FloatTensor)
			ac+= labels.size(0)
        
				#if (batch_id)%200 == 0:
					#accuracy = torch.sum(torch.max(output, 1)[1].cuda().data == labels).type(torch.FloatTensor)/labels.size(0)
					
				#	print(batch_id,z.data.cpu().numpy(),accuracy)
		print('train',c/ac)
		ac= 0.0
		c = 0.0
		cf = np.array([[0]*5]*5)
		loss = 0.0
		for a in valloader:
			vx,vy = a
			vx = vx.cuda()
			vy = vy.cuda()
			out = net(vx)
			loss += nn.functional.cross_entropy(out,vy)/vy.size(0)
			c+=torch.sum(torch.max(out, 1)[1].cuda().data == vy).type(torch.FloatTensor)
			ac+= vy.size(0)
			cf += metrics.confusion_matrix(torch.max(out, 1)[1].cuda().data.cpu().numpy(),vy.cpu().numpy(),labels = list(range(5)))
		loss/= c
		print('Valid epoch:',ep,'loss',loss.data.cpu().numpy(),'accuracy',c/ac)
		print(cf)
		if loss < best_loss:
			best_loss = loss
		#	open(save_path+'best.pt','w').close()
		#	open(save_path+'best.pt.lr','w').close()
			torch.save(net.state_dict(), save_path+'best.pt')
			torch.save(opt.state_dict(),save_path+'best.pt.lr')
		#open(save_path+str(ep)+'.pt','w').close()
		#open(save_path+str(ep)+'.pt.lr','w').close()
		#saving model and optimizer
		torch.save(net.state_dict(), save_path+str(ep)+'.pt')
		torch.save(opt.state_dict(),save_path+str(ep)+'.pt.lr')

import sys
if __name__ == '__main__':
	train(0.001,50,128)

