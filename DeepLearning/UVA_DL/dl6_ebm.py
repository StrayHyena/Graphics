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

mnist_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
dataset = torchvision.datasets.MNIST(DATASET_PATH,True,transform=mnist_transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset,BATCH_SIZE,shuffle=True,drop_last=True)

class MNISTEBM(nn.Module):
    def __init__(self,d_hidden=32,sample_buffer_size=10000):
        super().__init__()
        d1,d2,d3 = d_hidden//2,d_hidden,d_hidden*2
        # in: (1,28,28)
        self.net = nn.Sequential(
            nn.Conv2d(1,d1,kernel_size=3, stride=2,padding=1),  # out: (d1,14,14)
            nn.SiLU(),
            nn.Conv2d(d1,d2,kernel_size=3, stride=2,padding=2), # out: (d2,8,8)
            nn.SiLU(),
            nn.Conv2d(d2,d3,kernel_size=3, stride=2,padding=1), # out: (d3,4,4)
            nn.SiLU(),
            nn.Conv2d(d3,d3,kernel_size=3, stride=2,padding=1), # out: (d3,2,2)
            nn.SiLU(),
            nn.Flatten(),  
            nn.Linear(d3*2*2,d3*2),
            nn.SiLU(),
            nn.Linear(d3*2,1)
        )
        samples = torch.rand((sample_buffer_size,1,28,28))*2-1
        self.register_buffer('samples',samples)
    def forward(self,x):return self.net(x).squeeze(-1)
    @property
    def save_path(self):return os.path.join(MODEL_DIR,type(self).__name__)
    def Sample(self,cnt=BATCH_SIZE,step_size=10,std=0.005,K=60,sample_buffer_rate=0.95):
        prev_enable_grad,prev_training,GRAD_CLIP,device = torch.is_grad_enabled(),self.training,0.03,self.samples.device
        torch.set_grad_enabled(True)
        self.eval()
        for p in self.parameters():p.requires_grad=False
        # Langevin MCMC 注意 np.where的b要是(cnt,1,28,28)(这是cnt个不同的噪声)而不能是(1,1,28,28)(这是被广播的cnt个相同的噪声)
        choice = torch.randint(0,len(self.samples),(cnt,))
        x = torch.where((torch.rand(cnt,device=device)<sample_buffer_rate).reshape(cnt,1,1,1), self.samples[choice], torch.rand((cnt,1,28,28),device=device)*2-1).requires_grad_()  # x0
        for _ in range(K):
            self.forward(x).sum().backward() 
            x = (x.detach() + std*torch.randn(x.shape,device=device) - step_size*x.grad.clamp_(-GRAD_CLIP,GRAD_CLIP)).clamp_(-1,1).requires_grad_()

        torch.set_grad_enabled(prev_enable_grad) # restore previous enable grad
        if prev_training:self.train()
        for p in self.parameters(): p.requires_grad_()
        self.samples[choice] = x.detach()
        return x.detach()

def Train(energy_model,epoch_num=100,alpha=0.1):
    shutil.rmtree(LOG_PATH, ignore_errors=True);os.makedirs(LOG_PATH, exist_ok=True)
    optimizer = optim.AdamW(energy_model.parameters(),lr=1e-4,betas=(0.0,0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
    energy_model.to(DEVICE)
    for epoch in tqdm.tqdm(range(epoch_num)):
        energy_model.train()
        epoch_L,epoch_Lcdiv,epoch_Lreg,epoch_posmean,epoch_negmean, = 0.0,0.0,0.0,0.0,0.0,                                   
        for x_pos,_ in dataloader:
            x_pos.add_(torch.randn_like(x_pos) * 0.005).clamp_(min=-1.0, max=1.0) # 
            optimizer.zero_grad()
            E_pos,E_neg = energy_model(torch.cat([x_pos.to(DEVICE),energy_model.Sample()],dim=0)).chunk(2,dim=0)
            Lcdiv,Lreg =E_pos.mean()-E_neg.mean(), alpha*(E_pos*E_pos+E_neg*E_neg).mean()
            L = Lcdiv+Lreg
            L.backward()
            torch.nn.utils.clip_grad_norm_(energy_model.parameters(), max_norm=0.1)
            optimizer.step()
            epoch_L += L.item();epoch_Lcdiv += Lcdiv.item();epoch_Lreg += Lreg.item();epoch_posmean += E_pos.mean().item();epoch_negmean += E_neg.mean().item();
        scheduler.step()    
        LOGGER.add_scalar('L',epoch_L,epoch)
        LOGGER.add_scalar('LCD',epoch_Lcdiv,epoch)
        LOGGER.add_scalar('LREG',epoch_Lreg,epoch)
        LOGGER.add_scalar('avg+',epoch_posmean,epoch)
        LOGGER.add_scalar('avg-',epoch_negmean,epoch)
    torch.save(energy_model.state_dict(),energy_model.save_path)

def Main():
    model = MNISTEBM()
    if not os.path.exists(model.save_path):Train(model)
    model.load_state_dict(torch.load(model.save_path),strict=True)
    generated  = model.Sample(32,K=1024,sample_buffer_rate=0.0).cpu()
    im_grid = torchvision.utils.make_grid(torch.stack([generated[i] for i in range(32)],dim=0),nrow=8,padding=2,normalize=True,value_range=(-1,1))
    plt.imshow(im_grid.permute(1,2,0))
    plt.show()

if __name__ == '__main__': Main()