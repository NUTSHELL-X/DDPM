import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import os
from tqdm import tqdm
from dataset import create_dataloader
from utils import save_image_seq,save_weights,save_training_params
from options import config_parser
from model import UNet

parser=config_parser()
args=parser.parse_args()
h=args.h
w=args.w
dataloader=create_dataloader((h,w),args.batch_size,args.dataset_type)
num_steps = args.num_steps

betas = torch.linspace(-10, 10, num_steps)
betas = torch.sigmoid(betas)*(2e-2 - 1e-4)+1e-4

alphas = 1-betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_log = torch.log(1-alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1-alphas_prod)

print("alphas_bar_sqrt[-1]:",alphas_bar_sqrt[-1])
assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
    alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape == one_minus_alphas_bar_sqrt.shape

if args.device == "cuda" and torch.cuda.is_available():
    device = "cuda:" + str(args.gpus[0])
else:
    device = "cpu"

print('using device {device}'.format(device=device))

total_epochs = 0
model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
model=nn.DataParallel(model,device_ids=args.gpus)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def q_x(x_0, t): # q process , gradually adding noise to image
    noise = torch.randn_like(x_0).to(x_0.device)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t*x_0+alphas_1_m_t*noise)

def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    # print("x_0",x_0.shape)
    batch_size = x_0.shape[0]
    t = torch.randint(0, n_steps, size=(batch_size,))
    t = t.unsqueeze(-1)

    a = alphas_bar_sqrt[t][:,:,None,None].to(x_0.device)
    am1 = one_minus_alphas_bar_sqrt[t][:,:,None,None].to(x_0.device)
    e = torch.randn_like(x_0).to(x_0.device)
    x = x_0*a+e*am1
    output = model(x, t.squeeze(-1))

    loss = F.mse_loss(output,e)
    return loss


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, device):
    cur_x = torch.randn(shape).to(device)
    x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
        x_seq.append(cur_x)
    return x_seq


def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    coeff = betas[t]/one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t)
    x = x.to("cpu")
    eps_theta = eps_theta.to("cpu")
    mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt()
    sample = mean+sigma_t*z

    return (sample)

def save_noise_seq(image,num_steps,interval,save_folder): # show the process of adding noise
    add_noise_seq = []
    for i in range(num_steps//interval):
        add_noise_seq.append(q_x(image,i*interval).cpu())
    save_image_seq(add_noise_seq,save_folder,image_name_prefix="noise_seq")

def train():
    for i in range(args.epochs):
        print('updating epoch: {}'.format(total_epochs+i))
        for idx,(images,labels) in tqdm(enumerate(dataloader)):
            optimizer.zero_grad()
            images = images.to(device)
            loss = diffusion_loss_fn(model, images, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        
        print("loss:",loss)
        save_image_seq(images=images.detach().cpu(),save_folder="real",image_name_prefix="real_{}".format(total_epochs+i))
        with torch.no_grad():
            x_seq = p_sample_loop(model, images[0:1].shape, num_steps, betas, one_minus_alphas_bar_sqrt)

        for step in range(1, num_steps//20):
            cur_x = x_seq[step*20].detach().cpu()
            save_image(cur_x,os.path.join(args.generated_image_folder,f'generated_img_{i}_{step}.jpg'))


if __name__ == "__main__":
    if args.save_images:
        if not os.path.exists(args.generated_image_folder):
            os.makedirs(args.generated_image_folder,exist_ok=True)
    if args.continues:
        # load model weights
        model.module.load_state_dict(torch.load(args.model_weights_path))
        print("loaded weights from {}".format(args.model_weights_path))
        # load optimizer state and trained epochs
        training_params=torch.load(args.training_params_path)
        optimizer.load_state_dict(training_params['opt_state_dict'])
        total_epochs=training_params['total_epochs']
        print('total_epochs:',total_epochs)
    else:
        total_epochs=0
        print('initializing from scratch')
    # train 
    train()

    total_epochs += args.epochs

    save_weights(model,args.model_weights_path)
    save_training_params(optimizer,total_epochs,args.training_params_path)