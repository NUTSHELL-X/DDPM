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
num_steps = args.num_steps
betas = torch.linspace(1e-4, 2e-2, num_steps)

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

model = UNet(
        T=num_steps, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
model=nn.DataParallel(model,device_ids=args.gpus)


def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
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

if __name__=="__main__":
    # load model weights
    model.module.load_state_dict(torch.load(args.model_weights_path))
    print("loaded weights from {}".format(args.model_weights_path))
    model.eval()

    if not os.path.exists(args.generated_image_folder_test):
        os.makedirs(args.generated_image_folder_test,exist_ok=True)

    interval = 20
    img_num = 10
    assert num_steps >= interval

    for i in range(img_num):
        print(f"generating image {i}")
        with torch.no_grad():
            x_seq = p_sample_loop(model, (1,3,h,w), num_steps, betas, one_minus_alphas_bar_sqrt)
        # save generating process images
        for step in range(0, num_steps//interval):
            cur_x = x_seq[step*interval].detach().cpu()
            save_image(cur_x,os.path.join(args.generated_image_folder_test,f'generated_img_{i}_{step*interval}.jpg'))
        # save final generated image
        save_image(x_seq[-1],os.path.join(args.generated_image_folder_test,f'generated_img_{i}_{num_steps-1}.jpg'))