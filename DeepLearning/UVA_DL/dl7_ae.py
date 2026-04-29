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

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513 ,0.26158784))
])
train_set_,test_set = torchvision.datasets.CIFAR10(DATASET_PATH,True,transform=transform, download=True),torchvision.datasets.CIFAR10(DATASET_PATH,False,transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_set_, [45000, 5000])
train_loader,val_loader,test_loader = torch.utils.data.DataLoader(train_set,BATCH_SIZE,shuffle=True,drop_last=True),torch.utils.data.DataLoader(val_set,BATCH_SIZE,shuffle=True,drop_last=False),torch.utils.data.DataLoader(test_set,BATCH_SIZE,shuffle=True,drop_last=False)

class AutoEncoder(nn.Module):
    class Encoder(nn.Module):
        def __init__(self,hidden_dim,latent_dim):
            super().__init__()
            self.net = nn.Sequential(     
                nn.Conv2d(3,hidden_dim//2,3,stride=2, padding=1),  # out 16,16
                nn.GELU(),
                nn.Conv2d(hidden_dim//2,hidden_dim,3,stride=2,padding=1),  # out 8,8  
                nn.GELU(),
                nn.Conv2d(hidden_dim,hidden_dim*2,3,stride=2,padding=1),  # out 4,4  
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(16*hidden_dim*2,latent_dim),   
            )
        def forward(self,x):return self.net(x)
    class Decoder(nn.Module):
        def __init__(self,hidden_dim,latent_dim):
            super().__init__()
            self.linear = nn.Sequential(nn.Linear(latent_dim,16*hidden_dim*2),nn.GELU())
            self.net = nn.Sequential(     
                nn.ConvTranspose2d(hidden_dim*2,hidden_dim,3,2,1,output_padding=1),  # out 8,8
                nn.GELU(),
                nn.Conv2d(hidden_dim,hidden_dim,3,stride=1, padding=1),  # same
                nn.GELU(),
                nn.ConvTranspose2d(hidden_dim,hidden_dim//2,3,2,1,output_padding=1),  # out 16,16
                nn.GELU(),
                nn.Conv2d(hidden_dim//2,hidden_dim//2,3,stride=1,padding=1),  # same
                nn.GELU(),
                nn.ConvTranspose2d(hidden_dim//2,3,3,2,1,output_padding=1),  # out 32,32
                nn.GELU(),
                nn.Conv2d(3,3,3,stride=1,padding=1),  # same  
            )
        def forward(self,x):return self.net(self.linear(x).reshape(x.size(0),-1,4,4))
    def __init__(self, hidden_dim=32,latent_dim=384):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = self.Encoder(hidden_dim,latent_dim)
        self.decoder = self.Decoder(hidden_dim,latent_dim)
    def forward(self,x):return self.decoder(self.encoder(x))
    @property
    def save_path(self):return os.path.join(MODEL_DIR,type(self).__name__+'-'+str(self.latent_dim))

def Train(model,epoch_num=500):
    shutil.rmtree(LOG_PATH, ignore_errors=True);os.makedirs(LOG_PATH, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(),weight_decay=4e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2,patience=20,min_lr=5e-5)
    model.to(DEVICE)
    for epoch in tqdm.tqdm(range(epoch_num)):
        train_loss,val_loss = 0.0,0.0
        model.train()
        for batch_imgs,_ in train_loader:
            batch_imgs = batch_imgs.to(DEVICE)
            optimizer.zero_grad()
            loss = F.mse_loss(model(batch_imgs),batch_imgs, reduction='mean')
            loss.backward()
            optimizer.step()
            train_loss+=loss.detach().item()
        LOGGER.add_scalar('train_loss',train_loss/len(train_loader),epoch)
        # compute val loss
        model.eval()
        with torch.no_grad():
            for batch_imgs,_ in val_loader:
                batch_imgs = batch_imgs.to(DEVICE)
                val_loss += F.mse_loss(model(batch_imgs),batch_imgs, reduction='sum').detach().item()
            scheduler.step(val_loss/len(val_loader.dataset))
        LOGGER.add_scalar('val_loss',val_loss/len(val_loader.dataset),epoch)
    torch.save(model.state_dict(),model.save_path)

def Main():
    model = AutoEncoder()
    if not os.path.exists(model.save_path):Train(model)
    model.load_state_dict(torch.load(model.save_path,weights_only=True),strict=True)
    model.eval().to(DEVICE)
    
    print('Rebuild images using auto-encoder. First rebuild training data, then test data...\n\n')
    N = 10
    for dataset in (train_set,test_set):
        imgs = torch.stack([dataset[i][0] for i in torch.randint(0,len(dataset),(N,))])
        rebuild_imgs = model(imgs.to(DEVICE)).detach().cpu()
        im_grid = torchvision.utils.make_grid(torch.cat([imgs,rebuild_imgs],dim=0),nrow=N,padding=2,normalize=True,value_range=(-1,1))
        plt.imshow(im_grid.permute(1,2,0))
        plt.show()
    print('Find visually similar images.\n For each of the N randomly selected test images, retrieve the top-K most similar images from the training set.')
    N,K = 4,10
    query = torch.stack([test_set[i][0] for i in torch.randint(0,len(test_set),(N,))]).to(DEVICE) # (N,3,32,32)
    imgs = torch.cat([batch_imgs for batch_imgs,_ in train_loader],dim=0).to(DEVICE)
    latent_imgs,latent_query = model.encoder(imgs),model.encoder(query)
    dist = torch.cdist(latent_query,latent_imgs).squeeze()  # (N,latent_dim)  (N_train,latent_dim)  ==> (N,N_train)
    _,neighbor_idx = torch.topk(dist,K,dim=1, largest=False) 
    neighbor_imgs = imgs[neighbor_idx]  # (N,K,3,32,32)
    divider = torch.ones((N, 1, 3, 32, 32)).to(DEVICE)
    im_grid = torchvision.utils.make_grid(torch.cat([query.unsqueeze(1),divider,neighbor_imgs],dim=1).cpu().reshape(-1,3,32,32),nrow=K+2,padding=2,normalize=True,value_range=(-1,1))
    plt.imshow(im_grid.permute(1,2,0))
    plt.show()

if __name__ == '__main__': Main()