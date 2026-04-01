import torch,tqdm,os,shutil
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42) # Setting the seed
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs',os.path.splitext(os.path.basename(__file__))[0])
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models',os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(log_path, ignore_errors=True) 
os.makedirs(log_path, exist_ok=True)
if not os.path.exists(model_path): os.makedirs(model_path)

class XORDataset(data.Dataset):
    def __init__(self,size,radius = 0.1):
        super().__init__()
        self.size = size
        self.data  = torch.randint(0, 2, (size,2))
        self.label = (self.data[:,0] ^ self.data[:,1]).long() # 异或运算
        self.data = self.data.float() + (torch.rand(size,2)* 2 - 1) * radius  
    def __len__(self): return self.size
    def __getitem__(self, idx):return self.data[idx], self.label[idx]

class XORClassifier(nn.Module):
    def __init__(self,input_num = 2,hidden_num = 4,output_num = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_num, hidden_num)  # param number = 2*4 + 4 = 12
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_num, output_num) # param number = 4*1 + 1 = 5
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

def Train(model, dataloader, criterion, optimizer, num_epochs=100):
    writer = SummaryWriter(log_path)
    # model.train()  
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_loss = 0.0
        for batch_idx,(batch_datas,batch_lables) in enumerate(dataloader):
            if epoch==0 and batch_idx==0: writer.add_graph(model, batch_datas)
            outputs = model(batch_datas)
            loss = criterion(outputs, batch_lables.float().unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        writer.add_scalar('training_loss',epoch_loss,global_step = epoch + 1)

def Test(model, dataloader):
    # model.eval()
    correct = 0
    for batch_datas, batch_labels in dataloader:
        outputs = model(batch_datas)
        predicted = (torch.sigmoid(outputs) > 0.5).long()
        correct += (predicted.squeeze() == batch_labels).sum().item()
    return correct/len(dataloader.dataset)

def Main():
    model = XORClassifier()
    # print("model's total parameter number is  ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_dataloader = data.DataLoader(XORDataset(2500), batch_size=128, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # 注意只有对于那些train/test流程不同的网络(比如有dropout(),batchnorm()),train(),test()才会起作用
    Train(model, train_dataloader, nn.BCEWithLogitsLoss(), optimizer)
    print(Test(model,data.DataLoader(XORDataset(1000), batch_size=128, shuffle=True)))

Main()