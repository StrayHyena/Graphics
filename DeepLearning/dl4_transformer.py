import torch,tqdm,os,shutil,enum,math,torchvision
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision import transforms
from rich.traceback import install;install()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device=='cpu': print('device is cpu!')
torch.manual_seed(42) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs',os.path.splitext(os.path.basename(__file__))[0])  # ./logs/filename/
logger = SummaryWriter(log_path)
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models',os.path.splitext(os.path.basename(__file__))[0]) # ./models/filename/
if not os.path.exists(model_dir): os.makedirs(model_dir)
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'datasets') # ./datasets
batch_size = 64

class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self,optimizer, warmup_idx,iter_num):
        self.warmup_idx,self.iter_num = warmup_idx,iter_num
        super().__init__(optimizer)
    def get_lr(self):
        factor = 0.5*(1+np.cos(np.pi*self.last_epoch/self.iter_num))
        if self.last_epoch<=self.warmup_idx:factor*=self.last_epoch/self.warmup_idx
        return [factor*base_lr for base_lr in self.base_lrs]

class Transformer(nn.Module):
    class PositionalEncoding(nn.Module):
        def __init__(self,T,dm): # T:sequence length  dm: embedding's dimension
            super().__init__()
            pe = torch.zeros(T,dm)
            # 这里可以优化，使用tensor的广播机制向量化操作。                    
            for j in range(T):
                for i in range(dm):
                    pe[j][i] = np.sin(j/10000**(i/dm)) if i%2==0 else  np.cos(j/10000**((i-1)/dm))
            self.register_buffer('pe',pe)
        def forward(self,x): return x+self.pe  # auto broadcasting
    # note:下面注释里写的shape都省去了Batch size.实际执行代码的时候是有这一维度的
    class MultiHeadAttention(nn.Module):
        def __init__(self,dm,h=1):  # T:sequence length dm:embedding dim h:head number
            super().__init__()
            self.dk,self.W = dm//h,nn.ModuleDict()
            for s in ('Q','K','V'): self.W[s] = nn.ModuleList( [nn.Linear(dm,self.dk,bias=False) for i in range(h)] )
        def forward(self,x):  # x shape (T,dm)
            out = []
            for Wq,Wk,Wv in zip(self.W['Q'],self.W['K'],self.W['V']):
                # print(x.device,next(Wq.parameters()).device)
                Q,K,V = Wq(x),Wk(x),Wv(x)                                      # shape (T,dk)
                A = nn.Softmax(dim=-1)(Q@K.transpose(-1,-2)/math.sqrt(self.dk)) # shape (T,T)
                out.append(A@V)                                                # shape (T,dk)
            return torch.cat(out,dim=-1)  # shape (T,dm)

    class Encoder(nn.Module):
        def __init__(self,dm,h=1,dh=-1): # @dh: hidden dimension of the feed forward layer; @h: head number 
            super().__init__()
            if dh == -1: dh=4*dm
            self.mha = Transformer.MultiHeadAttention(dm,h)
            self.ln_mha = nn.LayerNorm(dm)
            self.ff = nn.Sequential(
                nn.Linear(dm,dh),
                nn.ReLU(),
                nn.Linear(dh,dm),
            )
            self.ln_ff = nn.LayerNorm(dm)
        def forward(self,x): # output  shape: (T,dm);|  input(embedding(after positional encoding)) shape: (T,dm); 
            y = self.ln_mha(x+self.mha(x))
            return self.ln_ff(y + self.ff(y))  
    
    # an encoder only transformer  
    def __init__(self,T,dm,h,d_out,dh=-1,encoder_num=6,pe=True,name=''):
        super().__init__()
        self.T,self.pe = T,Transformer.PositionalEncoding(T,dm) if pe else None
        self.encoders = nn.Sequential(*(Transformer.Encoder(dm,h,dh)for _ in range(encoder_num)))
        self.name=type(self).__name__+name+('_without'if self.pe is None else '_with')+'_pe'
        self.out_net = nn.Sequential(
            nn.LayerNorm(dm),
            nn.ReLU(),
            nn.Linear(dm,d_out),
        )
    def forward(self,x):return self.out_net(self.encoders(self.pe(x)if self.pe is not None else x))  # output shape: (T,d_out)
    @property
    def save_path(self):return os.path.join(model_dir,self.name)

# @EXPERIMENT0 这个测试里，我们的任务是把一个长度是T的有序整数[0,128)序列s,进行反转(e.g.(2,3,6,7,1) --> (1,7,6,3,2))
# 首先，考虑数字的embedding: 这里可以直接用 one-hot编码。而transformer的输出是T个logits
# 这里的label还是T个数字,使用CrossEntropyLoss可以直接和T个logits做loss。而在Eval里,这T个logits返算出的tag必须和label里的数字一一对应
class SequenceReverse:
    name = __qualname__
    class DataSet(data.Dataset):
        def __init__(self,T=10,ed=128,n=20000):
            super().__init__()
            self.numclass,self.data = ed,torch.randint(ed,(n,T))
        def __len__(self):return self.data.shape[0]
        def __getitem__(self,i):return F.one_hot(self.data[i],self.numclass).float(),self.data[i].flip(dims=[-1])
    @staticmethod
    def Test():
        T,dm = 10,128
        train_set,val_set,test_set = SequenceReverse.DataSet(T,dm),SequenceReverse.DataSet(T,dm,n=1000),SequenceReverse.DataSet(T,dm,n=10000),
        train_loader,val_loader,test_loader = data.DataLoader(train_set,batch_size,drop_last=True),data.DataLoader(train_set,batch_size),data.DataLoader(train_set,batch_size)
        model = Transformer(T,dm,1,train_set.numclass,name=SequenceReverse.name,pe=True)     
        if not os.path.exists(model.save_path): Train(model,train_loader,val_loader)
        model.load_state_dict(torch.load(model.save_path),strict=True)
        print(model.name + ' test acc is ',Eval(model,test_loader))
        test_x = torch.tensor( (0,1,4,65,77,12,34,65,4,100)).to(device)
        test_y = model(F.one_hot(test_x,dm).float().unsqueeze(0)) # (1,T,dm)
        print('Inference Test\n x:',test_x,'\n y:',test_y.squeeze().argmax(dim=-1))

# @EXPERIMENT1 这个测试里，我们的任务是从一个长度是T的图片集合中找出类别不同的那一张。e.g.(0:猫,1:狗,2:猫,3:猫,4:猫,) --> 1
#首先，需要对每个图片都做embedding，这里我们使用提前预训练好的模型(i.e. ResNet34),ResNet34最后的fc会把维度从512-->1000，这里我们把fc换成identity,得到512维的embedding。
# Transformer输出的是T个1维tensor。所以Train里有squeeze(dim=-1)
# 这里的label是1个数字(即,num class,范围是0-T),使用CrossEntropyLoss可以直接和已经squeeze的size==T的tensor做loss。
class SetAnormalyDetection:
    name = __qualname__
    class DataSet(data.Dataset):
        def __init__(self,T=10,train=True):
            super().__init__()
            file_path = os.path.join(dataset_path,SetAnormalyDetection.name+'-'+('Train' if train else 'Test')+'.pth')
            if not os.path.exists(file_path):
                from torchvision.datasets import CIFAR100
                # ImageNet statistics. Note,DO NOT USE CIFAR100's MEAN & STD
                DATA_MEANS = np.array([0.485, 0.456, 0.406])
                DATA_STD = np.array([0.229, 0.224, 0.225])
                transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(DATA_MEANS,DATA_STD)])
                data_set = CIFAR100(root=dataset_path, train=train, transform=transform, download=True)
                embeddings,targets = [],[] # imgs
                # test_set = CIFAR100(root=dataset_path, train=False, transform=transforms,download=True)
                res34 = torchvision.models.resnet34(pretrained=True).to(device)
                res34.fc = nn.Identity().to(device) # 原始的fc是 512->1000;现在就直接是512
                res34.eval()
                with torch.no_grad():
                    for imgs,labels in data.DataLoader(data_set,64,shuffle=False,drop_last=False):
                        embeddings.append(res34(imgs.to(device)))  # (B,X)
                        targets.append(labels.to(device))
                embeddings,targets = torch.cat(embeddings,dim=0),torch.cat(targets,dim=0)
                label2embeddings,datas,labels = [embeddings[targets==k] for k in range(100)],[],[]
                for _ in range(data_set.data.shape[0]):
                    norm_label,anorm_label = np.random.choice(100,2,False) # choose two distinct label
                    anorm_idx,anorm_img_idx = np.random.randint(0,T),np.random.randint(0,label2embeddings[anorm_label].shape[0])# 从anorm里选1张
                    seqimgs = label2embeddings[norm_label][np.random.choice(label2embeddings[norm_label].shape[0],T,False)]  # 从norm里挑选T个image
                    seqimgs[anorm_idx] = label2embeddings[anorm_label][anorm_img_idx] # 从anorm里选1张
                    datas.append(seqimgs)
                    labels.append(anorm_idx)
                torch.save({'datas':torch.stack(datas),'labels':torch.tensor(labels)},file_path)
            loaded = torch.load(file_path)
            self.datas,self.labels = loaded['datas'],loaded['labels']
            self.dm = self.datas.shape[-1]
        def __len__(self):return self.labels.shape[0]
        # data (T,512)  label (1,)
        def __getitem__(self,i): return self.datas[i],self.labels[i]
    @staticmethod
    def Test():
        T = 10
        train_val_set,test_set = SetAnormalyDetection.DataSet(T),SetAnormalyDetection.DataSet(T,False)
        train_set,val_set = torch.utils.data.random_split(train_val_set,[45000,5000])
        train_loader,val_loader,test_loader = data.DataLoader(train_set,batch_size,True,drop_last=True),data.DataLoader(val_set,batch_size,True,drop_last=False),data.DataLoader(test_set,batch_size,True,drop_last=False)
        model = Transformer(T,test_set.dm,8,1,pe=False,name = SetAnormalyDetection.name)
        if not os.path.exists(model.save_path):Train(model,train_loader,val_loader,num_epochs=20)
        model.load_state_dict(torch.load(model.save_path),strict=True)
        print(model.name + ' test acc is ',Eval(model,test_loader))

@torch.no_grad()
def Eval(model,dataloader):
    model,correct,num = model.to(device),0,0
    model.eval()
    for seqs,labels in dataloader:
        seqs,labels = seqs.to(device),labels.to(device)
        predict = model(seqs) # (B, T, number class) for SeqReverse ,  (B,T,1) for SetAnormal
        correct += (predict.squeeze(dim=-1).argmax(dim=-1)==labels).sum().item()
        num += labels.size(0) * (predict.shape[1] if SequenceReverse.name in model.name else 1)  # batch size * number classes 
    return correct/num

def Train(model,train_loader,val_loader,criterion=nn.CrossEntropyLoss(),num_epochs=10):
    shutil.rmtree(log_path, ignore_errors=True) 
    os.makedirs(log_path, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(),weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer,50,num_epochs*len(train_loader))
    model,best_val_acc = model.to(device),0.0
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_loss = 0.0
        model.train()
        for batch_idx,(seqs,labels) in enumerate(train_loader):
            y_hat = model(seqs.to(device)).squeeze(dim=-1)
            # 如果 y_hat shape是(B,T,num_class)最后一维是class维度，但是CrossEntropyLoss要求，input的Class维度在dim=1(第二位)
            # 如果 y_hat shape是(B,num_class) 就无需transpose
            if len(y_hat.shape)==3: y_hat = y_hat.transpose(-1,-2)
            loss = criterion(y_hat,labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss+=loss.item()*len(labels)
        logger.add_scalar(f'{model.name} train loss',epoch_loss,global_step=epoch)
        val_acc = Eval(model,val_loader)
        logger.add_scalar(f'{model.name} val acc',val_acc,global_step=epoch)
        if val_acc>best_val_acc         :
            best_val_acc=val_acc
            torch.save(model.state_dict(),model.save_path)

def Main():
    for TestType in [ SetAnormalyDetection,    SequenceReverse,]: TestType().Test()
if __name__ == '__main__': Main()