import torch,tqdm,os,shutil,enum,math,torchvision
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
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

# NOTE!  这里的transform很重要！！！
mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(), # 除以255,原始PIL的像素值范围[0,255]
    lambda x: (x*255).to(torch.int32)
])
dataset = torchvision.datasets.MNIST(DATASET_PATH,True,transform=mnist_transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset,BATCH_SIZE,shuffle=True,drop_last=True)

class Mask:
    @staticmethod
    def CheckerBoard(switch=0):
        x0 = next(iter(dataloader))[0][0]
        maski,maskj =  torch.arange(x0.shape[-2]).unsqueeze(1),torch.arange(x0.shape[-1]).unsqueeze(0)
        return ((maski+maskj)%2==switch).float()
    @staticmethod
    def Channel(c_in,switch=0):
        assert c_in%2==0
        mask = torch.cat([torch.zeros(c_in//2),torch.ones(c_in//2)]).float().reshape(1,c_in,1,1)
        return (switch-mask).abs()

class Dequantization(nn.Module):
    def __init__(self,alpha=1e-5,quants=256):
        super().__init__()
        self.alpha,self.quants = alpha,quants
    # x shape (B,C,H,W);  for not reverse : NOTE, Expect x's value in range [0,255] (NOT [0,1] as usual)
    def forward(self,x,ldj,reverse=False):
        n = x.numel()/x.shape[0]
        if not reverse:
            x = x.float()
            x = x + torch.rand_like(x).detach()

            ldj += -n*np.log(self.quants)
            x = x / self.quants
            ldj += n*np.log(1-self.alpha)
            x = 0.5*self.alpha+(1-self.alpha)*x
            # x = logit(x)
            ldj += (-x.log()-(1-x).log()).sum(dim=[1,2,3])
            x = x.log()-(1-x).log()
        else:
            ldj += (-x-2*F.softplus(-x)).sum(dim=[1,2,3])
            x = F.sigmoid(x)
            
            ldj += -n*np.log(1-self.alpha)
            x = (x-self.alpha/2)/(1-self.alpha)
            
            ldj += n*np.log(self.quants)
            x = x * self.quants

            x = x.floor().clamp(0,self.quants-1).to(torch.int32)
        return x,ldj

class CouplingLayer(nn.Module):
    class ContactELU(nn.Module):
        def __init__(self): super().__init__()
        def forward(self,x):return torch.cat([F.elu(x),F.elu(-x)],dim=1) # double channel dimension
    class LayerNormChannel(nn.Module):
        def __init__(self,c):
            super().__init__()
            self.mean,self.std = nn.Parameter(torch.zeros((1,c,1,1))),nn.Parameter(torch.ones((1,c,1,1)))
        def forward(self,x):
            x = (x-x.mean(dim=1,keepdim=True))/(x.var(dim=1,keepdim=True)+1e-5).sqrt()
            return x*self.std+self.mean
    class GatedConvNet(nn.Module):
        class GatedConv(nn.Module):
            def __init__(self,c_in,c_hidden): 
                super().__init__()
                self.net = nn.Sequential(
                    CouplingLayer.ContactELU(),
                    nn.Conv2d(2*c_in,c_hidden,3,1,1),
                    CouplingLayer.ContactELU(),
                    nn.Conv2d(2*c_hidden,2*c_in,3,1,1),
                )
            def forward(self,x):
                x0,x1 = self.net(x).chunk(2,dim=1)
                return x + x0*x1.sigmoid()
        
        def __init__(self,c_in,c_hidden,layer_num=3):
            super().__init__()
            layers = [nn.Conv2d(c_in,c_hidden,3,1,1)]
            for _ in range(layer_num):
                layers.append(CouplingLayer.GatedConvNet.GatedConv(c_hidden,c_hidden))
                layers.append(CouplingLayer.LayerNormChannel(c_hidden))
            layers.append(CouplingLayer.ContactELU())
            layers.append(nn.Conv2d(2*c_hidden,c_in*2,3,1,1))
            layers[-1].weight.data.zero_()
            layers[-1].bias.data.zero_()
            self.net = nn.Sequential(*layers)
        def forward(self,x):return self.net(x)  # out: (B,2C,H,W)

    def __init__(self,c_in,c_hidden,mask):
        super().__init__()
        self.register_buffer('m',mask)
        self.alpha = nn.Parameter(torch.zeros((c_in)).reshape(1,c_in,1,1))
        self.net = CouplingLayer.GatedConvNet(c_in,c_hidden)
    def forward(self,x,ldj,reverse=False):
        s,t = self.net(x*self.m).chunk(2,dim=1)
        ea = self.alpha.exp()
        s = (s/ea).tanh()*ea
        s,t = s*(1-self.m),t*(1-self.m)
        if not reverse: x,ldj = x*s.exp()+t, ldj + s.sum(dim=[1,2,3])
        else: x,ldj = (x-t)/s.exp(), ldj - s.sum(dim=[1,2,3])
        return x,ldj

class Squeeze(nn.Module):
    def forward(self,x,ldj,reverse=False): 
        return F.pixel_unshuffle(x,2) if not reverse else F.pixel_shuffle(x,2) ,ldj

class Split(nn.Module):
    def forward(self,x,ldj,reverse = False):
        N = torch.distributions.Normal(0,1)
        if not reverse:
            x,split = x.chunk(2,dim=1)
            ldj = ldj + N.log_prob(split).sum(dim=[1,2,3])
        else:
            split = torch.randn_like(x)
            ldj = ldj - N.log_prob(split).sum(dim=[1,2,3])
            x = torch.cat([x,split],dim=1)
        return x,ldj

class ImageFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleList([
            Dequantization(),
            CouplingLayer(1,32,Mask.CheckerBoard(0)),
            CouplingLayer(1,32,Mask.CheckerBoard(1)),
            Squeeze(),
            CouplingLayer(4,48,Mask.Channel(4,0)),
            CouplingLayer(4,48,Mask.Channel(4,1)),
            Split(),
            Squeeze(),
            CouplingLayer(8,64,Mask.Channel(8,0)),
            CouplingLayer(8,64,Mask.Channel(8,1)),
            CouplingLayer(8,64,Mask.Channel(8,0)),
            CouplingLayer(8,64,Mask.Channel(8,1)),
        ])
        # last x shape is (B,8,7,7)
    def forward(self,x,reverse=False):
        ldj = torch.zeros((x.shape[0],),device=x.device)
        for net in (self.nets if not reverse else reversed(self.nets)):  
            x,ldj = net(x,ldj,reverse) 
        return x,ldj
    @property
    def save_path(self):return os.path.join(MODEL_DIR,type(self).__name__)

def Train(model,epoch_num=200):
    shutil.rmtree(LOG_PATH, ignore_errors=True);os.makedirs(LOG_PATH, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(),lr=1e-3,betas=(0.0,0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
    N = torch.distributions.Normal(0,1)
    model.to(DEVICE)
    for epoch in tqdm.tqdm(range(epoch_num)):
        model.train()
        epoch_loss = 0.0
        for batch_imgs,_ in dataloader:
            optimizer.zero_grad()
            z,ldj = model(batch_imgs.to(DEVICE))
            loss = -(ldj+N.log_prob(z).sum(dim=[1,2,3])).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            epoch_loss+= BATCH_SIZE*loss.item()
        LOGGER.add_scalar('epoch_loss',epoch_loss/len(dataset),epoch)
        LOGGER.add_scalar('bpd', epoch_loss/len(dataset)*np.log2(np.e)/(28*28),epoch)
        scheduler.step()    
    torch.save(model.state_dict(),model.save_path)

def SanityCheck(model):
    ldj,test_x = torch.ones((3,)),next(iter(dataloader))[0][:3]
    test_y,ldj = model(test_x,ldj,False)
    test_x_,ldj = model(test_y,ldj,True)
    print(np.allclose(test_x_.detach().numpy(),test_x.detach().numpy()),ldj)

def Main():
    model = ImageFlow()
    if not os.path.exists(model.save_path):Train(model)
    model.load_state_dict(torch.load(model.save_path),strict=True)
    x,ldj = model.to(DEVICE)(torch.randn((16,8,7,7),device=DEVICE),True)
    im_grid = torchvision.utils.make_grid(x.cpu(),nrow=4,padding=2)
    plt.imshow(im_grid.permute(1,2,0))
    plt.show()

if __name__ == '__main__': Main()