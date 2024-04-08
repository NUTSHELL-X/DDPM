import configargparse
import torch

def config_parser():
    parser=configargparse.ArgumentParser()
    parser.add_argument('--config',is_config_file=True,help='config file path',default='config.yaml')
    parser.add_argument('--device',type=str,default='cuda')
    parser.add_argument('--model_weights_path',type=str) # path to save weights(.pt file)
    parser.add_argument('--training_params_path',type=str) # path to save training parameters(optimizor,total training epochs)
    parser.add_argument('--save_images',type=bool,default=True) # whether save images while training or not
    parser.add_argument('--generated_image_folder',type=str) # path to save generated images(.jpg file)
    parser.add_argument('--generated_image_folder_test',type=str) # path to save generated images when testing(.jpg file)
    parser.add_argument('--dataset_path',type=str) # path containing dataset
    parser.add_argument('--dataset_type',type=str,default='image_folder') # dataset type (see dataset.py)
    parser.add_argument('--num_steps',type=int,default=1000) # T
    parser.add_argument('--epochs',type=int) # training epochs
    parser.add_argument('--dtype',type=str) # data type used for training(float16,float32)
    parser.add_argument('--h',type=int) # height of image
    parser.add_argument('--w',type=int) # width of image
    parser.add_argument('--batch_size',type=int) # batch size
    parser.add_argument('--lr',type=float) # learning rate
    parser.add_argument('--lr_decay',type=bool,default=False) # learning rate decay or not
    parser.add_argument('--lr_step_size',type=int,default=10) # lr decay frequency
    parser.add_argument('--milestones',type=int,nargs='+') # milestones for learning rate decay
    parser.add_argument('--gamma',type=float) # learning rate decay ratio
    parser.add_argument('--continues',type=bool,default=False) # continues training or not
    parser.add_argument('--gpus',nargs='+',type=int,default=0) # used gpus e.g [0,1]

    return parser

if __name__=='__main__': # show all args
    parser=config_parser()
    args=parser.parse_args()
    print(args)
    for arg in vars(args):
        value = getattr(args,arg)
        print('arg: ',arg,'value: ',value,'type: ',type(value))