import torch,tqdm,os,shutil,enum,math,torchvision,torch_geometric,torch_scatter
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,numpy as np,torch_geometric.nn as geom_nn, torch_geometric.data as geom_data
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torch_geometric.utils import add_self_loops
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
    def __init__(self,head_num,d_in,d_out,dropout,last_layer=True):  
        super().__init__()
        self.h,self.last_layer,d_prime = head_num,last_layer,d_out
        if not last_layer:  assert d_out % head_num == 0;d_prime //= head_num
        self.W = nn.ModuleList([nn.Linear(d_in,d_prime)for _ in range(self.h)])
        self.a = nn.ModuleList([nn.Linear(2*d_prime,1)for _ in range(self.h)])
        self.attn_sigma,self.attn_dropout = nn.LeakyReLU(0.2),nn.Dropout(dropout)
    # ACHTUNG: MAKE SURE edge_index contains self loop already!
    def forward(self,x,edge_index): # x(N,d_in)  edge_index(2,E)
        N,x_tildes = x.size(-2), []
        v_src,v_tgt = edge_index  # (E) (E)
        for W,a in zip(self.W,self.a):
            Wx = W(x)  # (N,d')
            WxPair =  torch.concat([Wx[v_tgt],Wx[v_src]],dim=-1)  # Wx[v_tgt] and Wx[v_src] shape (E,d'); result shape (E,2d')
            alpha = torch_scatter.scatter_softmax(self.attn_sigma(a(WxPair)).squeeze(-1),v_tgt,0,N)  # (E)
            alpha = self.attn_dropout(alpha)
            x_tildes.append(torch_scatter.scatter_add(alpha.unsqueeze(-1)*Wx[v_src],v_tgt,dim=0,dim_size=N))   # alpha.unsqu(E,1)  * Wx[vi] (E,d')
        x_primes = torch.stack(x_tildes)  # (h,N,d')
        return x_primes.mean(0) if self.last_layer else x_primes.permute(1,0,2).reshape(N,-1)

class NodeLevelTask:
    class NodeClassifier(nn.Module):
        def __init__(self,d_in,class_num,layer_num,head_num=8,d_hidden=64,dropout=0.6):
            super().__init__()
            self.name = type(self).__name__
            layers = [GATLayer(head_num,d_in,d_hidden,dropout,last_layer=False),nn.ELU(),nn.Dropout(dropout)]
            for _ in range(layer_num-2): layers.extend([GATLayer(head_num,d_hidden,d_hidden,dropout,last_layer=False),nn.ReLU(),nn.Dropout(dropout)])
            layers.append(GATLayer(head_num,d_hidden,class_num,dropout,last_layer=True))
            self.layers = nn.ModuleList(layers)
        def forward(self,x,edge_index):  # x (N,d_in)  ACHTUNG:NO Batch dim
            for layer in self.layers:
                if isinstance(layer,GATLayer): x = layer(x,edge_index)
                else: x = layer(x)
            return x  # (N,class_num)
        @property
        def save_path(self):return os.path.join(MODEL_DIR,self.name)
    @torch.no_grad()
    def Eval(model,data,mask):
        model,data,mask = model.to(DEVICE),data.to(DEVICE),mask.to(DEVICE)
        model.eval()
        y_ = model(data.x,data.edge_index) # (N,class_num)
        return (y_.argmax(dim=-1)[mask]==data.y[mask]).sum().item()/mask.sum().item()
    def Train(model,data,epoch_num = 100):
        shutil.rmtree(LOG_PATH, ignore_errors=True) 
        os.makedirs(LOG_PATH, exist_ok=True)
        optimizer = optim.AdamW(model.parameters(),weight_decay=5e-4)
        model,data = model.to(DEVICE),data.to(DEVICE)
        x,y,edge_index,train_mask,val_mask,test_mask = data.x,data.y,data.edge_index,data.train_mask,data.val_mask,data.test_mask
        criterion,best_val_acc = nn.CrossEntropyLoss(), 0.0
        for epoch in tqdm.tqdm(range(epoch_num)):
            model.train()
            y_ = model(x,edge_index)
            optimizer.zero_grad()
            loss = criterion(y_[train_mask],y[train_mask])
            loss.backward()
            optimizer.step()
            train_acc = (y_.argmax(dim=-1)[train_mask]==data.y[train_mask]).sum().float().item()/train_mask.sum().float().item()
            val_acc = NodeLevelTask.Eval(model,data,val_mask)
            if val_acc>best_val_acc : best_val_acc = val_acc; torch.save(model.state_dict(),model.save_path)
            LOGGER.add_scalar(f'{model.name} loss ',loss,global_step=epoch)
            LOGGER.add_scalar(f'{model.name} val_acc ',val_acc,global_step=epoch)
            LOGGER.add_scalar(f'{model.name} train_acc ',train_acc,global_step=epoch)
    def Run():
        # cora里只有一张图 len(cora)==1,无需使用dataloader |x(N,d_in词袋) ,y(N论文类别), edge_index ,train_mask,test_mask,val_mask; 
        cora = torch_geometric.datasets.Planetoid(root=DATASET_PATH, name="Cora",transform=torch_geometric.transforms.AddSelfLoops())
        model = NodeLevelTask.NodeClassifier(cora[0].x.shape[-1],     cora[0].y.max().item()+1,   2)
        if not os.path.exists(model.save_path): NodeLevelTask.Train(model,cora[0])
        model.load_state_dict(torch.load(model.save_path),strict=True)
        print(model.name,' test acc is ',NodeLevelTask.Eval(model,cora[0],cora[0].test_mask))

class GraphLevelTask:
    class GraphClassifier(nn.Module):
        def __init__(self,d_in,class_num,layer_num,head_num=8,d_hidden=256,dropout_gat=0.0,dropout_linear = 0.5):
            super().__init__()
            self.name = type(self).__name__
            layers = [GATLayer(head_num,d_in,d_hidden,dropout_gat,last_layer=False),nn.ELU(),nn.Dropout(dropout_gat)]
            for _ in range(layer_num-2): layers.extend([GATLayer(head_num,d_hidden,d_hidden,dropout_gat,last_layer=False),nn.ReLU(),nn.Dropout(dropout_gat)])
            layers.append(GATLayer(head_num,d_hidden,d_hidden,dropout_gat,last_layer=True))
            self.layers = nn.ModuleList(layers) 
            self.out = nn.Sequential(nn.Dropout(dropout_linear),nn.Linear(d_hidden,class_num))
        def forward(self,x,edge_index,batch_idx):  # x (N,d_in)  batch_index (N)
            for layer in self.layers:
                if isinstance(layer,GATLayer): x = layer(x,edge_index)
                else: x = layer(x)
            x = torch_geometric.nn.global_mean_pool(x,batch_idx)
            return  self.out(x)  # (B,class_num)
        @property
        def save_path(self):return os.path.join(MODEL_DIR,self.name)
    @torch.no_grad()
    def Eval(model,dataloader):
        model,acc = model.to(DEVICE),0
        model.eval()
        for data in dataloader:
            data = data.to(DEVICE)  
            x,y,edge_index,batch_index = data.x,data.y,data.edge_index,data.batch
            y_ = model(x,edge_index,batch_index)
            acc += (y_.argmax(dim=-1)==y).sum().item()
        return acc/len(dataloader.dataset)
    def Train(model,dataloader,epoch_num = 100):
        shutil.rmtree(LOG_PATH, ignore_errors=True) 
        os.makedirs(LOG_PATH, exist_ok=True)
        optimizer = optim.AdamW(model.parameters(),weight_decay=5e-4)
        model,criterion = model.to(DEVICE),nn.CrossEntropyLoss()
        for epoch in tqdm.tqdm(range(epoch_num)):
            model.train()
            epoch_loss,epoch_train_acc = 0.0,0.0
            for data in dataloader:
                data = data.to(DEVICE)  
                x,y,edge_index,batch_index = data.x,data.y,data.edge_index,data.batch
                y_ = model(x,edge_index,batch_index)
                optimizer.zero_grad()
                loss = criterion(y_,y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss*len(data.y)
                epoch_train_acc += (y_.argmax(dim=-1)==y).sum().item()
            LOGGER.add_scalar(f'{model.name} loss ',epoch_loss/len(dataloader.dataset),global_step=epoch)
            LOGGER.add_scalar(f'{model.name} train_acc ',epoch_train_acc/len(dataloader.dataset),global_step=epoch)
        torch.save(model.state_dict(),model.save_path)
    def Run():
        #实际上, mutag.x mutag.edge_index 已经是188张小图拼接的大图了
        mutag = torch_geometric.datasets.TUDataset(root=DATASET_PATH, name="MUTAG",transform=torch_geometric.transforms.AddSelfLoops())
        # 以batch为大小拼接小图
        train_loader,test_loader = torch_geometric.data.DataLoader(mutag[:150],BATCH_SIZE,True),torch_geometric.data.DataLoader(mutag[150:],BATCH_SIZE,True)
        model = GraphLevelTask.GraphClassifier(mutag[0].x.shape[-1],2,3)
        if not os.path.exists(model.save_path): GraphLevelTask.Train(model,train_loader,500)
        model.load_state_dict(torch.load(model.save_path),strict=True)
        print(model.name,' test acc is ',GraphLevelTask.Eval(model,test_loader))

def Main():
    # NodeLevelTask.Run()
    GraphLevelTask.Run()
if __name__ == '__main__': Main()