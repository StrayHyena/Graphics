import torch,tqdm,os,shutil,enum,math,torchvision
import torch.utils.data as data,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available(),torch.cuda.get_device_name(),'device is ',device)
torch.manual_seed(42) 
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'logs',os.path.splitext(os.path.basename(__file__))[0])
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'models',os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(log_path, ignore_errors=True) 
os.makedirs(log_path, exist_ok=True)
if not os.path.exists(model_dir): os.makedirs(model_dir)
dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'datasets')

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_dataset = FashionMNIST(root=dataset_path, train=True, transform=transform, download=True)
test_set = FashionMNIST(root=dataset_path, train=False, transform=transform, download=True)
train_set, val_set = torch.utils.data.random_split(train_dataset, [50000, 10000])
train_loader = data.DataLoader(train_set, batch_size=1024, shuffle=True, drop_last=False,pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=1024, shuffle=True, drop_last=False)

class Activation(nn.Module):
    class Type(enum.Enum):
        Sigmoid,Tanh,ReLU,LeakyReLU,ELU,Swish = range(6)
    def Sigmoid(self,x): return 1/(1+torch.exp(-x))
    def Tanh(self,x): return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))
    def ReLU(self,x): return x * (x > 0).float()
    def LeakyReLU(self,x, alpha=0.1): return torch.where(x > 0, x, x * alpha)
    def ELU(self,x, alpha=1.0): return torch.where(x > 0, x, alpha * (torch.exp(x) - 1))
    def Swish(self,x): return x * self.Sigmoid(x)

    def __init__(self,type=Type.ReLU):
        super().__init__()
        self.type = type
    def forward(self,x): return self.Act(self.type)(x)
    def Act(self,type):
        if type==Activation.Type.Sigmoid: return self.Sigmoid
        if type==Activation.Type.Tanh: return self.Tanh
        if type==Activation.Type.ReLU: return self.ReLU
        if type==Activation.Type.LeakyReLU: return self.LeakyReLU
        if type==Activation.Type.ELU: return self.ELU
        if type==Activation.Type.Swish: return self.Swish
    def Plot(self):
        fig,ax_ = plt.subplots(2,3,figsize=(12,8))
        for act_type,ax in enumerate(ax_.flatten()):
            x = torch.linspace(-5,5,1000,requires_grad=True)
            ax.set_xlim(-5,5)
            y = self.Act(Activation.Type(act_type))(x)
            y.sum().backward()
            ax.set_title(Activation.Type(act_type).name)
            ax.plot(x.detach().numpy(),y.detach().numpy(),label='f')
            ax.plot(x.detach().numpy(),x.grad.detach().numpy(),label='∇f')
            ax.legend()
        plt.show()

class Net(nn.Module):
    def __init__(self, act_type, input_size=784, num_classes=10, hidden_sizes=[512, 256, 256, 128],batch_norm = False):
        super().__init__()
        layer_sizes,layers,name = [input_size,*hidden_sizes], [],act_type.name
        for i in range(1,len(layer_sizes)):
            layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
            if batch_norm:layers.append(nn.BatchNorm1d(layer_sizes[i]))
            layers.append(Activation(act_type))
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        self.net = nn.Sequential(*layers) # 只有当self.xxx=某个nn.Module时，才会被加入到模型的参数中
    def forward(self,x):return self.net(x)
    def PrintParameterInfo(self):
        for name,param in self.named_parameters():
            print(name,param.numel())  # 注意len(param)只能得到第一维的长度

def GetModel(act_type,batch_norm = False):
    model_name = act_type.name+("_bn" if batch_norm else "")
    model_path = model_dir + f'/{model_name}.pth'
    def TrainAndSave(model, dataloader, criterion, optimizer, num_epochs=100):
        writer,accuracy = SummaryWriter(log_path),0.0
        model.train()  
        for epoch in tqdm.tqdm(range(num_epochs)):
            epoch_loss = 0.0
            for batch_idx,(batch_datas,batch_lables) in enumerate(dataloader):
                batch_datas = batch_datas.reshape(batch_datas.size(0), -1).to(device)
                batch_lables = batch_lables.to(device)
                if epoch==0 and batch_idx==0: writer.add_graph(model, batch_datas)
                outputs = model(batch_datas)
                loss = criterion(outputs, batch_lables)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()/len(dataloader)
            writer.add_scalar( model_name + ' training_loss',epoch_loss,global_step = epoch + 1)
            val_accuracy = Test(model, val_loader) 
            writer.add_scalar( model_name + ' val_accuracy',accuracy,global_step = epoch + 1)
            if accuracy< val_accuracy:
                accuracy = val_accuracy
                torch.save(model.state_dict(),model_path)
    if not os.path.exists(model_path):  
        net = Net(act_type, batch_norm=batch_norm).to(device)
        TrainAndSave(net,train_loader,F.cross_entropy,optim.SGD(net.parameters(),0.01,0.9))
    model = Net(act_type,batch_norm=batch_norm)
    model.load_state_dict(torch.load(model_path), strict=True)
    print('load model ',model_path)
    return model.to(device)

def Test(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch_datas, batch_labels in dataloader:
            batch_datas = batch_datas.reshape(batch_datas.size(0), -1).to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_datas)
            correct += (torch.argmax(outputs,dim=1) == batch_labels).sum().item()
    return correct/len(dataloader.dataset)

def Main(plot_activation = False,show_some_dataset = False,
         plot_gradient_statics = False,show_accuracy = True):
    if plot_activation: Activation().Plot()
    if show_some_dataset: 
        # note that data_set[i] is data,label pair, so data_set[i][0] is the data
        im_grid = torchvision.utils.make_grid(torch.stack([data_set[i][0] for i in range(16)],dim=0),nrow=4,padding=2,normalize=True)
        plt.imshow(im_grid.permute(1,2,0))
        plt.show()
    if plot_gradient_statics:
        test_data = data.DataLoader(train_set, batch_size=256, shuffle=True)
        images,labels = next(iter(test_data))
        images = images.reshape(images.size(0), -1)
        grad_data = {} # {str(activation name):{str(parameter name):numpy array of gradients}}
        for i in range(6):
            model = Net(Activation.Type(i))
            model.zero_grad()
            predicts = model(images)
            loss = F.cross_entropy(predicts, labels)
            loss.backward()
            grad_data[Activation.Type(i).name] = {name:param.grad.detach().numpy().reshape(-1) for name,param in model.named_parameters() if 'weight' in name }
        model_num,named_param_num,cmap = len(grad_data),len( next(iter(grad_data.values())) ),plt.get_cmap("tab10") 
        fig,ax = plt.subplots(model_num,named_param_num, figsize=(named_param_num, model_num))
        for model_i,(model_name,grad_info) in enumerate(grad_data.items()):
            ax[model_i][0].set_ylabel(model_name)
            for param_i,(param_name,grads) in enumerate(grad_info.items()):
                sns.histplot(grads,bins=30,ax = ax[model_i][param_i],kde=True,color = cmap(model_i))
                ax[model_i][param_i].ticklabel_format(style='plain', axis='x') # 关闭x轴的科学计数法
                ax[0][param_i].set_title(param_name)
                ax[-1][param_i].set_xlabel('gradient value')
        fig.subplots_adjust(wspace=0.45)
        plt.show()
    if show_accuracy:
        for i in range(6):
            model = GetModel(Activation.Type(i), batch_norm=False)
            print(Activation.Type(i).name, ' accuracy: ', Test(model, data.DataLoader(test_set, batch_size=256)))

if __name__ == "__main__":
    Main(plot_activation=False,show_some_dataset=False,plot_gradient_statics=False,show_accuracy= True)