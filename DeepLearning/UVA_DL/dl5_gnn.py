import torch,tqdm,os,shutil,enum,math,torchvision,torch_geometric
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,numpy as np,torch_geometric.nn as geom_nn, torch_geometric.data as geom_data
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision import transforms
from rich.traceback import install;install()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE=='cpu': print('DEVICE is cpu!')
torch.manual_seed(42) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs',os.path.splitext(os.path.basename(__file__))[0])  # ./logs/filename/
LOGGER = SummaryWriter(LOG_PATH)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models',os.path.splitext(os.path.basename(__file__))[0]) # ./models/filename/
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'datasets') # ./datasets
BATCH_SIZE = 64

class GATLayer(nn.Module):
    def __init__(self,head_num,d_in,d_out,is_concat=True):  
        super().__init__()
        self.h,self.is_concat,d_prime = head_num,is_concat,d_out
        if is_concat:  assert d_out % head_num == 0;d_prime //= head_num
        self.W = nn.ModuleList([nn.Linear(d_in,d_prime)for _ in range(self.h)])
        self.a = nn.ModuleList([nn.Linear(2*d_prime,1)for _ in range(self.h)])
        self.attn_sigma,self.sigma,self.softmax = nn.LeakyReLU(0.2),nn.ELU(),nn.Softmax(dim=-1)
    # ACHTUNG MAKE SURE edge_index contains self loop already!
    def forward(self,x,edge_index): # x(N,d_in)  edge_index(2,E)
        N,x_tildes = x.size(-2), []
        vi,vj = edge_index  # (E) (E)
        for W,a in zip(self.W,self.a):
            Wx = W(x)  # (N,d')
            WxPair =  torch.concat([Wx[vi],Wx[vj]],dim=-1)  # Wx[vi] and Wx[vj] shape (E,d'); result shape (E,2d')
            yPair = self.attn_sigma(a(WxPair))  # (E,1)
            A = torch.zeros(N,N)
            A.scatter_add_(-1,vj,yPair)
            x_tildes.append(self.softmax(A)@Wx)   # (N,d')
        x_primes = torch.stack(x_tildes)  # (h,N,d')
        x_primes = x_primes.permute(1,0,2).reshape(N,-1) if self.is_concat else x_primes.mean(0)
        return self.sigma(x_primes)

class NodeLevelTask:
    @staticmethod
    def Run():
        # cora里只有一张图,无需使用dataloader
        # x(N,d_in) ,y(N), edge_index,train_mask,test_mask,val_mask
        cora = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora")
        

@torch.no_grad()
def Eval(model,dataloader):
    model,correct,num = model.to(DEVICE),0,0
    model.eval()
    for seqs,labels in dataloader:
        seqs,labels = seqs.to(DEVICE),labels.to(DEVICE)
        predict = model(seqs) # (B, T, number class) for SeqReverse ,  (B,T,1) for SetAnormal
        correct += (predict.squeeze(dim=-1).argmax(dim=-1)==labels).sum().item()
        num += labels.size(0) * (predict.shape[1] if SequenceReverse.name in model.name else 1)  # batch size * number classes 
    return correct/num

def Train(model,train_loader,val_loader,criterion=nn.CrossEntropyLoss(),num_epochs=10):
    shutil.rmtree(LOG_PATH, ignore_errors=True) 
    os.makedirs(LOG_PATH, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(),weight_decay=1e-4)
    scheduler = CosineWarmupScheduler(optimizer,50,num_epochs*len(train_loader))
    model,best_val_acc = model.to(DEVICE),0.0
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_loss = 0.0
        model.train()
        for batch_idx,(seqs,labels) in enumerate(train_loader):
            y_hat = model(seqs.to(DEVICE)).squeeze(dim=-1)
            # 如果 y_hat shape是(B,T,num_class)最后一维是class维度，但是CrossEntropyLoss要求，input的Class维度在dim=1(第二位)
            # 如果 y_hat shape是(B,num_class) 就无需transpose
            if len(y_hat.shape)==3: y_hat = y_hat.transpose(-1,-2)
            loss = criterion(y_hat,labels.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss+=loss.item()*len(labels)
        LOGGER.add_scalar(f'{model.name} train loss',epoch_loss,global_step=epoch)
        val_acc = Eval(model,val_loader)
        LOGGER.add_scalar(f'{model.name} val acc',val_acc,global_step=epoch)
        if val_acc>best_val_acc         :
            best_val_acc=val_acc
            torch.save(model.state_dict(),model.save_path)

def Main():
    NodeLevelTask.Run()
if __name__ == '__main__': Main()