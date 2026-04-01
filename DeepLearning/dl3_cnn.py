import torch,tqdm,os,shutil,enum,math,torchvision
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,seaborn as sns,numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision import transforms
# from rich.traceback import install;install()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print('device is ',device)
torch.manual_seed(42) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs',os.path.splitext(os.path.basename(__file__))[0])  # ./logs/filename/
logger = SummaryWriter(log_path)
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models',os.path.splitext(os.path.basename(__file__))[0]) # ./models/filename/
shutil.rmtree(log_path, ignore_errors=True) 
os.makedirs(log_path, exist_ok=True)
if not os.path.exists(model_dir): os.makedirs(model_dir)
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'datasets') # ./datasets
batch_size = 128

RGBMEAN,RGBSTD = (0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513 ,0.26158784)
train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((32,32), scale=(0.8,1.0), ratio=(0.9,1.1)),
                transforms.ToTensor(),
                transforms.Normalize(RGBMEAN, RGBSTD),
                                     ])
test_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(RGBMEAN, RGBSTD)])
train_dataset = CIFAR10(root=dataset_path, train=True, download=True,transform=train_transform)
test_set = CIFAR10(root=dataset_path, train=False, download=True,transform=test_transform)
train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
_, val_set = torch.utils.data.random_split(train_dataset, [45000, 5000])
train_loader,val_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True), data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False)
test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=False)
# shape is  (B,C,H,W)  50000,3,32,32

@torch.no_grad()
def Eval(model,dataloader):
    model,correct=model.to(device),0
    model.eval()
    for images,labels in dataloader:
        images,labels = images.to(device),labels.to(device)
        predict = model(images) # (batch count, number class)
        correct += (predict.argmax(dim=1)==labels).sum().item() 
    # NOTE: len(dataloader) is the number of batches!!!
    return correct/len(dataloader.dataset)

def Train(model,criterion=nn.CrossEntropyLoss(),optimizer='sgd',num_epochs=180):
    opt = optim.SGD(model.parameters(),0.1,0.9,weight_decay=1e-4)
    if optimizer.lower()=='adam':opt = optim.AdamW(model.parameters(),weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[100, 150],  gamma=0.1)
    model,best_val_acc = model.to(device),0.0
    for epoch in tqdm.tqdm(range(num_epochs)):
        epoch_loss = 0.0
        model.train()
        for batch_idx,(images,labels) in enumerate(train_loader):
            loss = criterion(model(images.to(device)),labels.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss+=loss.item()*len(labels)
        scheduler.step()
        logger.add_scalar(f'{model.name} train loss',epoch_loss,global_step=epoch)
        val_acc = Eval(model,val_loader)
        logger.add_scalar(f'{model.name} val acc',val_acc,global_step=epoch)
        if val_acc>best_val_acc         :
            best_val_acc=val_acc
            torch.save(model.state_dict(),model.save_path)

class GoogleNet(nn.Module):
    # ~~~~~ STRUCTURE OF GOOGLE NET ~~~~~ 
    # Pre-Process : conv 
    # Inceptions  : inception(×2)-max pool(w/2,h/2)-inception(×4)-max pool(w/2,h/2)-inception(×2) 
    # Post-Process: avg - linear
    class Inception(nn.Module):
        # ~~~~~ STRUCTURE OF INCEPTION BLOCK ~~~~~ 
        # input(c_in)|[path0] ------------->             1x1 conv (c_out0) -|            
        #            |[path1] -> 1x1 conv (c_hidden0) -> 3x3 conv (c_out1) -|
        #            |[path2] -> 1x1 conv (c_hidden1) -> 5x5 conv (c_out2) -|
        #            |[path3] -> 3x3 maxpool          -> 1x1 conv (c_out3) -| ==> output(c_out0+c_out1+c_out2+c_out3)
        def __init__(self,c_in,c_out,c_hidden,act_fn=nn.ReLU):
            super().__init__()
            self.path0 = nn.Sequential(
                nn.Conv2d(c_in,c_out[0],kernel_size=1),
                nn.BatchNorm2d(c_out[0]),
                act_fn(),
            )
            self.path1 = nn.Sequential(
                nn.Conv2d(c_in,c_hidden[0],kernel_size=1),
                nn.BatchNorm2d(c_hidden[0]),
                act_fn(),
                nn.Conv2d(c_hidden[0],c_out[1],kernel_size=3,padding='same'),
                nn.BatchNorm2d(c_out[1]),
                act_fn(),
            )
            self.path2 = nn.Sequential(
                nn.Conv2d(c_in,c_hidden[1],kernel_size=1),
                nn.BatchNorm2d(c_hidden[1]),
                act_fn(),
                nn.Conv2d(c_hidden[1],c_out[2],kernel_size=5,padding='same'),
                nn.BatchNorm2d(c_out[2]),
                act_fn(),
            )
            self.path3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, padding=1,stride=1),
                nn.Conv2d(c_in, c_out[3], kernel_size=1),
                nn.BatchNorm2d(c_out[3]),
                act_fn(),
            )
        def forward(self,x): return torch.cat([self.path0(x),self.path1(x),self.path2(x),self.path3(x)],dim=1)

    def __init__(self,act_fn=nn.ReLU):
        super().__init__()
        self.name = type(self).__name__
        self.save_path = os.path.join(model_dir,self.name)
        self.prologue = nn.Sequential(
                nn.Conv2d(3,64,kernel_size=3,padding='same'),
                nn.BatchNorm2d(64),
                act_fn(),
        )
        self.inceptions = nn.Sequential(
            GoogleNet.Inception(64,[16,32,8,8],[32,16],act_fn),     
            GoogleNet.Inception(64,[24,48,12,12],[32,16],act_fn), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),    
            GoogleNet.Inception(96,[24,48,12,12],[32,16],act_fn),     
            GoogleNet.Inception(96,[16,48,16,16],[32,16],act_fn),     
            GoogleNet.Inception(96,[16,48,16,16],[32,16],act_fn),     
            GoogleNet.Inception(96,[32,48,24,24],[32,16],act_fn),     
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),    
            GoogleNet.Inception(128,[32,64,16,16],[48,16],act_fn),     
            GoogleNet.Inception(128,[32,64,16,16],[48,16],act_fn),     
        )
        self.epilogue = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128,10),
        )
        # self._init_params()
    def forward(self,x): return self.epilogue(self.inceptions(self.prologue(x)))

class ResNet(nn.Module):
    class Block(nn.Module):
        def __init__(self,c_in,down_sampling=False):
            super().__init__()
            c_out = c_in if not down_sampling else 2*c_in
            self.F = nn.Sequential(
                nn.Conv2d(c_in,c_out,kernel_size=3,stride=2 if down_sampling else 1,padding=1,bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),
                nn.Conv2d(c_out,c_out,kernel_size=3,stride=1,padding=1,bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.W = nn.Conv2d(c_in,c_out,kernel_size=1,stride=2) if down_sampling else None
            self.relu = nn.ReLU(inplace=True)
        def forward(self,x):return self.relu((self.W(x) if self.W is not None else x) +self.F(x))

    def __init__(self):
        super().__init__()
        self.name = type(self).__name__
        self.save_path = os.path.join(model_dir,self.name) 
        self.net = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
 # res net block begin ================================
            ResNet.Block(16),
            ResNet.Block(16),
            ResNet.Block(16),

            ResNet.Block(16,True),
            ResNet.Block(32),
            ResNet.Block(32),
 
            ResNet.Block(32,True),
            ResNet.Block(64),
            ResNet.Block(64),
 # res net block end ================================
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,10)
        )
        # self._init_params()
    def forward(self,x):return self.net(x)

class DenseNet(nn.Module):
    class Transition(nn.Module):
        def __init__(self,c_in,theta):
            super().__init__()
            self.c_out = int(c_in*0.5)
            self.net = nn.Sequential(
                nn.BatchNorm2d(c_in),
                nn.ReLU(),
                nn.Conv2d(c_in,self.c_out,1,bias=False),
                nn.AvgPool2d(2,2)
            )
        def forward(self,x):return self.net(x)
    
    class Block(nn.Module):
        class Layer(nn.Module):
            def __init__(self,c_in,k):
                super().__init__()
                self.net = nn.Sequential(
                    nn.BatchNorm2d(c_in),
                    nn.ReLU(),
                    nn.Conv2d(c_in,2*k,1),
                    nn.BatchNorm2d(2*k),
                    nn.ReLU(),
                    nn.Conv2d(2*k,k,3,padding=1),
                )
            def forward(self,x):return torch.cat((self.net(x),x),dim=1) 

        def __init__(self,c_in,n,k):
            super().__init__()
            self.net = nn.Sequential(*[DenseNet.Block.Layer(c_in+i*k,k) for i in range(n)])
        def forward(self,x):return self.net(x)

    def __init__(self,blocknums=[6,6,6,6], k = 16,theta = 0.5):
        super().__init__()
        self.name = type(self).__name__
        self.save_path = os.path.join(model_dir,self.name) 
        
        self.input_net = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1,bias=False),
            # 这里没加 BN ReLU是因为紧接着block里做了。总之随着网络核心结构调整 
        )
        blocks,c_hidden = [],16
        for bidx,layer_n in enumerate(blocknums):
            blocks.append(DenseNet.Block(c_hidden,layer_n,k))
            c_hidden += layer_n*k
            if bidx==len(blocknums)-1:break
            blocks.append(DenseNet.Transition(c_hidden,theta))
            c_hidden = blocks[-1].c_out
        self.blocks = nn.Sequential(*blocks)
        self.output_net = nn.Sequential(
            nn.BatchNorm2d(c_hidden),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(c_hidden,10),
        )
    def forward(self,x):return self.output_net(self.blocks(self.input_net(x)))

def Main():
    for ModelType in [DenseNet,GoogleNet,ResNet,]:
        model = ModelType()
        if not os.path.exists(model.save_path): 
            Train(model,optimizer='adam' if model.name=='GoogleNet'else 'sgd')
        model.load_state_dict(torch.load(model.save_path),strict=True)
        print(model.name + ' test acc is ',Eval(model,test_loader))

class Utils:
    @staticmethod
    def PrintModelMemory(model):
        total_params = 0
        total_size_bytes = 0
        for name, param in model.named_parameters():
            num_params = param.numel()
            size_bytes = param.element_size() * num_params  # 每个元素字节数 * 元素数量
            total_params += num_params
            total_size_bytes += size_bytes
        print(f"Total size (MB): {total_size_bytes/1024**2:.4f}")

    @staticmethod
    def ShowSomeSample(dataset):
        strlabels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        im_grid = torchvision.utils.make_grid(torch.stack([dataset[i][0] for i in range(16)],dim=0),nrow=4,padding=2,normalize=True)
        for i in range(4):print([strlabels[ dataset[j][1]] for j in range(4*i,4*i+4)])
        plt.imshow(im_grid.permute(1,2,0))
        plt.show()

if __name__ == '__main__': Main()