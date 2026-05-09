import torch,tqdm,os,shutil,torchvision,zipfile,math,json
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from rich.traceback import install;install()

DEVICE,RAND_SEED = torch.device("cuda" if torch.cuda.is_available() else "cpu"),43
if DEVICE=='cpu': print('DEVICE is cpu!')
torch.manual_seed(RAND_SEED) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(RAND_SEED)
    torch.cuda.manual_seed_all(RAND_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs',os.path.splitext(os.path.basename(__file__))[0])  # ./logs/filename/
LOGGER = SummaryWriter(LOG_PATH)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models',os.path.splitext(os.path.basename(__file__))[0]) # ./models/filename/
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),'datasets') # ./datasets
BATCH_SIZE = 64

# 尝试使用FGSM和Adversarial Patches去攻击在ImageNet数据集训练的ResNet34模型。  我们的play ground的数据集是自制的TinyImageNet
with zipfile.ZipFile(os.path.join(DATASET_PATH,'TinyImageNet.zip'), 'r') as zip_ref: zip_ref.extractall(os.path.join(DATASET_PATH))
DATA_MEAN,DATA_STD =  [0.485, 0.456, 0.406],[0.229, 0.224, 0.225]
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(DATA_MEAN, DATA_STD)
])
dataset = torchvision.datasets.ImageFolder(os.path.join(DATASET_PATH,'TinyImageNet'),transform)
dataloader = torchvision.utils.data.DataLoader(dataset,BATCH_SIZE,drop_last=True)
with open(os.path.join(DATASET_PATH,'TinyImageNet','label_list.json'), "r") as f:text_label = json.load(f)
model = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).to(DEVICE).eval()
for p in model.parameters():p.requires_grad_(False)

@torch.no_grad()
def EvaluateAccuracy(k):
    correct, total = 0, 0
    for imgs,labels in torch.utils.data.DataLoader(dataset,BATCH_SIZE,drop_last=True):
        imgs,labels = imgs.to(DEVICE),labels.to(DEVICE)
        _,topk_idx = torch.topk(model(imgs),k)
        correct += (topk_idx==labels.reshape(-1,1).expand_as(topk_idx)).sum().item()
        total+=labels.size(0)
    return correct/total

def AdversarialPatch(target='goldfish',size=64,epoch_num=10):
    patch,attack_label = nn.Parameter(torch.zeros((3,size,size),device=DEVICE)), torch.ones((BATCH_SIZE),device=DEVICE)*text_label.index(target)
    optimizer = optim.AdamW([patch])
    for epoch in tqdm.tqdm(range(epoch_num)):
        for imgs,_ in dataloader:
            imgs = imgs.to(DEVICE)
            # apply patch
            for img in imgs:
                offset_x,offset_y = np.random.randint(0,imgs.shape[-1]-size),np.random.randint(0,imgs.shape[-2]-size)
                img[:,offset_x:offset_x+size,offset_y+size] = (torch.tanh(patch)+1-2*DATA_MEAN)/(2*DATA_STD)
            optimizer.zero_grad()
            F.cross_entropy(model(imgs),attack_label).backward()
            optimizer.step()
        

def Main(n=20,k=5,attack_type='FGSM'):
    indices = torch.randint(0,len(dataset),(n,))
    imgs = torch.stack([dataset[i][0] for i in indices])*torch.tensor(DATA_STD).reshape(1,3,1,1)+torch.tensor(DATA_MEAN).reshape(1,3,1,1)
    labels = torch.tensor([dataset[i][1] for i in indices])
    if attack_type == 'FGSM': 
        imgs = imgs.to(DEVICE).requires_grad_(True)
        F.cross_entropy(model(imgs),labels.to(DEVICE),reduction='sum').backward()
        imgs = imgs+0.02* torch.sign( imgs.grad)
    elif attack_type =='patch':imgs = AdversarialPatch(imgs,labels)
    topk_probs,topk_indices = torch.topk( torch.softmax(model(imgs.to(DEVICE)), dim=1),k)
    fig, axes = plt.subplots(4, 5, figsize=(20,10))
    for i,ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i].detach().cpu().permute(1, 2, 0))
        ax.set_title(f"True: {text_label[labels[i]]}", loc='left', fontsize=10)
        ax.axis("off")
        info_text,topkacc = "Top-k:\n",False
        for j in range(k):
            topkacc |= topk_indices[i][j]==labels[i]
            label_name = text_label[topk_indices[i][j].item()]
            prob_val = topk_probs[i][j].item()
            info_text += f"{label_name}\n   {prob_val:.2f};\n"
        ax.text(1.05, 0.5, info_text+'\n topk correct '+str(topkacc.item()), transform=ax.transAxes, ha='left', va='center', fontsize=7,bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
    plt.tight_layout() 
    plt.show()

if __name__ == '__main__': Main()