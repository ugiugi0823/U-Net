import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F

torch.manual_seed(0)
criterion = nn.BCEWithLogitsLoss() # loss
n_epochs = 200 
input_dim = 1 # 흑백이니까, 1dim
label_dim = 1 # 마차가지로  1dim
display_step = 20
batch_size = 4
lr = 0.0002
initial_shape = 512
target_shape = 373
device = 'cuda'





def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    # image_shifted = (image_tensor + 1) / 2
    image_shifted = image_tensor
    image_unflat = image_shifted.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=4)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()





# dataloader.py

from skimage import io
import numpy as np
volumes = torch.Tensor(io.imread('/content/C3W2A/train-volume.tif'))[:, None, :, :] / 255 # real
labels = torch.Tensor(io.imread('/content/C3W2A/train-labels.tif', plugin="tifffile"))[:, None, :, :] / 255 # label
# 생각에 요건 pix2pix 가 아니야, UNet 이야! 
labels = crop(labels, torch.Size([len(labels), 1, target_shape, target_shape]))
dataset = torch.utils.data.TensorDataset(volumes, labels)



def train():
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True)
    unet = UNet(input_dim, label_dim).to(device)
    unet_opt = torch.optim.Adam(unet.parameters(), lr=lr)
    cur_step = 0
    # vol = real = 실제 이미지, label = label = 세그맵(정답)
    # 그러니까 unet 은 label 에 가깝게 생성해야 하네
    # 그러면 실제 입력 값, 그러니까 내가 찾고 있는 인코더 거칠때 이미지가 작아져야 한는 친구는 vol 그러면 vol 이 어디로 이동하는지만 보면 될 거 같다.
    for epoch in range(n_epochs):
        for real, labels in tqdm(dataloader):
            cur_batch_size = len(real)
            # Flatten the image
            real = real.to(device)
            labels = labels.to(device)

            ### Update U-Net ###
            unet_opt.zero_grad()
            pred = unet(real)
            unet_loss = criterion(pred, labels)
            unet_loss.backward()
            unet_opt.step()

            if cur_step % display_step == 0:
                print(f"Epoch {epoch}: Step {cur_step}: U-Net loss: {unet_loss.item()}")
                show_tensor_images(
                    crop(real, torch.Size([len(real), 1, target_shape, target_shape])), 
                    size=(input_dim, target_shape, target_shape)
                )
                show_tensor_images(labels, size=(label_dim, target_shape, target_shape))
                show_tensor_images(torch.sigmoid(pred), size=(label_dim, target_shape, target_shape))
            cur_step += 1

train()