import torch,tqdm,os,shutil,enum,math,torchvision
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,numpy as np,torch_geometric.nn as geom_nn, torch_geometric.data as geom_data
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

class MNISTEnergy(nn.Module):
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
    def Sample(self,cnt=BATCH_SIZE,step_size=10,std=0.005,K=60):
        prev_enable_grad,prev_training,GRAD_CLIP = torch.enable_grad(),self.training,0.03
        torch.set_grad_enabled(True)
        self.eval()
        for p in self.parameters():p.requires_grad=False
        # Langevin MCMC
        x = torch.where(torch.rand(cnt)<0.05, torch.rand((1,28,28))*2-1, self.samples[:cnt]).requires_gard_()  # x0
        for _ in range(K):
            self.forward(x).sum().backward()  # now we have grad in x
            x += std*torch.randn(x.shape) - step_size*x.grad.clamp_(-GRAD_CLIP,GRAD_CLIP).detach()
            x.grad.zero_()
        
        torch.set_grad_enabled(prev_enable_grad) # restore previous enable grad
        if prev_training:self.train()
        for p in self.parameters(): p.requires_grad_()
        self.samples = torch.cat([self.samples[cnt:],x],dim=0)
        return x

def Train(energy_model,epoch_num=100):
    optimizer = optim.AdamW(energy_model.parameters(),weight_decay=5e-4,lr=1e-4)
    energy_model.to(DEVICE)
    for epoch in tqdm.tqdm(range(epoch_num)):
        energy_model.train()
        epoch_loss = 0.0
        for x_pos,_ in dataloader:
            optimizer.zero_grad()
            x_neg = energy_model.Sample()
            E_pos = energy_model(x_pos.to(DEVICE))
            E_neg = energy_model(x_neg.to(DEVICE))
            L = (E_pos.mean()-E_neg.mean()) + (E_pos*E_pos).mean()+(E_neg*E_neg).mean()
            L.backward()
            optimizer.step()
            epoch_loss += L
        LOGGER.add_scalar('loss',epoch_loss,epoch)
    torch.save(energy_model.state_dict(),energy_model.save_path)

def Main():
    model = MNISTEnergy()
    if not os.path.exists(model.save_path):Train(model)
    model.load_state_dict(torch.load(model.save_path),strict=True)

if __name__ == '__main__': Main()