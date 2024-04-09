# DDPM Pytorch Inplement
A pytorch DDPM inplementation

## Quick Start
### Training
First prepare your dataset.For simplest method, put your images(.jpg or .png format, rgb images) into data/images, or you can organize your images as pytorch image folder, and change dataset_path in config.yaml to your folder.
Then you can change options in config.yaml according to your preference, like batch_size, gpus, image resolutions etc.
After prepared data and set configs, just run "python train.py" to train your DDPM model, images generated after training each epoch will be saved in exp/generated as default.Model weights and optimizer state will be saved as exp/model.pth and exp/training_params.pt respectively after training is finished.
### Testing(Generating Images)

## Generated Examples

## Acknowledgements
UNet model used implementation of zoubohao:
https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-?tab=readme-ov-file