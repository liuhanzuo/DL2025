import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams


# figure size in inches optional
rcParams['figure.figsize'] = 11, 8

# read images
img_A = mpimg.imread('./ebm/groundtruth.png')
img_B = mpimg.imread('./ebm/corrupted.png')

# display images
fig, ax = plt.subplots(1, 2)
ax[0].imshow(img_A)
ax[1].imshow(img_B)

from utils import hello
hello()

from collections import deque
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from tqdm.autonotebook import tqdm

import numpy as np
import os
import time
import torch
import torch.nn as nn
import torchvision

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils import save_model, load_model, corruption, train_set, val_set

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

os.makedirs('./ebm', exist_ok=True)

class MlpBackbone(nn.Module):
    # feel free to modify this
    def __init__(self, input_shape, hidden_size, activation=nn.functional.silu):
        super(MlpBackbone, self).__init__()
        self.input_shape = input_shape  # (C, H, W)
        self.hidden_size = hidden_size
        # Layers
        self.fc1 = nn.Linear(np.prod(self.input_shape), self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, 1)

        self.activation = activation

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        out = self.fc4(x)
        return out
    
def langevin_step(energy_model, x, step_lr, eps, max_grad_norm):
    """
    Perform one step of Langevin dynamics sampling.

    Args:
        energy_model (nn.Module): The energy-based model used for sampling.
        x (torch.Tensor): The input tensor to update via Langevin dynamics.
        step_lr (float): The learning rate of the optimizer used to update the input.
        eps (float): The step size of the Langevin dynamics update.
        max_grad_norm (float or None): The maximum norm of the gradient for gradient clipping.

    Returns:
        torch.Tensor: The updated input tensor after one step of Langevin dynamics.
    """
    ##############################################################################
    #                  TODO: You need to complete the code here                  #
    ##############################################################################
    # YOUR CODE HERE
    y = x.detach().requires_grad_()
    if y.grad is not None:
        y.grad.zero_()
    energy_model2 = energy_model.__class__(input_shape=energy_model.input_shape, hidden_size=energy_model.hidden_size, activation=energy_model.activation).to(device)
    energy_model2.load_state_dict(energy_model.state_dict())
    energy_model2.zero_grad()
    energy = energy_model2(y).sum()
    energy.backward()
    grad = y.grad
    if torch.norm(grad) > max_grad_norm:
        grad *= (max_grad_norm/torch.norm(grad))
    x = x - step_lr * grad + torch.randn_like(x)*eps
    return x
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    
# Hyperparams
N_SAMPLE_STEPS = 100
L_STEP_NORM = 0.01
STEP_LR = 1000

def inpainting(energy_model, x, mask, n_steps, step_lr, max_grad_norm):
    """
    Inpainting function that completes an image given a masked input using Langevin dynamics.

    Args:
        energy_model (nn.Module): The energy-based model used to generate the image.
        x (torch.Tensor): The input tensor, a masked image that needs to be completed.
        mask (torch.Tensor): The mask tensor, with the same shape as x, where 1 indicates the corresponding
                             pixel is visible and 0 indicates it is missing.
        n_steps (int): The number of steps of Langevin dynamics to run.
        step_lr (float): The step size of Langevin dynamics.
        max_grad_norm (float or None): The maximum gradient norm to be used for gradient clipping. If None, 
                                       no gradient clipping is performed.

    Returns:
        torch.Tensor: The completed image tensor.
    """
    ##############################################################################
    #                  TODO: You need to complete the code here                  #
    ##############################################################################
    # YOUR CODE HERE
    for _ in range(n_steps):
        new_x = langevin_step(energy_model,x,step_lr,0.005,max_grad_norm)
        x = mask * x + (1 - mask) * new_x
        
    return x.clamp(0,1)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    
def evaluate(energy_model, val_loader, n_sample_steps, step_lr, langevin_grad_norm, device='cuda'):
    """
    Evaluates the energy model on the validation set and returns the corruption MSE,
    recovered MSE, corrupted images, and recovered images for visualization.

    Args:
        energy_model (nn.Module): Trained energy-based model.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        n_sample_steps (int): Number of Langevin dynamics steps to take when sampling.
        step_lr (float): Learning rate to use during Langevin dynamics.
        langevin_grad_norm (float): Maximum L2 norm of the Langevin dynamics gradient.
        device (str): Device to use (default='cuda').
    """
    mse = corruption_mse = 0
    energy_before_sampling = energy_after_sampling = 0
    n_batches = 0
    energy_model.eval()

    pbar = tqdm(total=len(val_loader.dataset))
    pbar.set_description('Eval')
    for data, _ in val_loader:
        n_batches += data.shape[0]
        data = data.to(device)
        broken_data, mask = corruption(data, type_='ebm')
        energy_before_sampling += energy_model(broken_data).sum().item()
        recovered_img = inpainting(energy_model, broken_data, mask,
                                   n_sample_steps, step_lr, langevin_grad_norm)
        energy_after_sampling += energy_model(recovered_img).sum().item()

        mse += np.mean((data.detach().cpu().numpy().reshape(-1, 28 * 28) - recovered_img.detach().cpu().numpy().reshape(-1, 28 * 28)) ** 2, -1).sum().item()
        corruption_mse += np.mean((data.detach().cpu().numpy().reshape(-1, 28 * 28) - broken_data.detach().cpu().numpy().reshape(-1, 28 * 28)) ** 2, -1).sum().item()

        pbar.update(data.shape[0])
        pbar.set_description('Corruption MSE: {:.6f}, Recovered MSE: {:.6f}, Energy Before Sampling: {:.6f}, Energy After Sampling: {:.6f}'.format(
            corruption_mse / n_batches, mse / n_batches, energy_before_sampling / n_batches, energy_after_sampling / n_batches))

    pbar.close()
    return (corruption_mse / n_batches, mse / n_batches, data[:100].detach().cpu(), broken_data[:100].detach().cpu(), recovered_img[:100].detach().cpu())

import random
def train(n_epochs, energy_model, train_loader, val_loader, optimizer, n_sample_steps, step_lr, langevin_eps, langevin_grad_norm, l2_alpha,
          device='cuda', buffer_maxsize=int(1e4), replay_ratio=0.95, save_interval=1):
    energy_model.to(device)
    replay_buffer = torch.zeros(buffer_maxsize, 1, 28, 28)
    buffer_size = buffer_ptr = 0
    best_mse = np.inf

    for epoch in range(n_epochs):
        train_loss = energy_before = energy_plus = energy_minus = n_batches = 0
        pbar = tqdm(total=len(train_loader.dataset))
        pbar.set_description('Train')
        for i, (x_plus, _) in enumerate(train_loader):
            n_batches += x_plus.shape[0]
            bs = x_plus.shape[0]

            # init negative samples
            if buffer_size == 0:
                x_minus = torch.rand_like(x_plus)
            else:
                ##############################################################################
                #                  TODO: You need to complete the code here                  #
                ##############################################################################
                # YOUR CODE HERE
                if random.random() < replay_ratio:
                    indices = (torch.rand([len(x_plus)]) * buffer_size).to(torch.int64)
                    x_minus = replay_buffer[indices[:]]
                else:
                    x_minus = torch.rand_like(x_plus)
                # raise NotImplementedError()
                ##############################################################################
                #                              END OF YOUR CODE                              #
                ##############################################################################
            x_minus = x_minus.to(device)

            energy_before += energy_model(x_minus).sum().item()

            # sample negative samples
            ##############################################################################
            #                  TODO: You need to complete the code here                  #
            ##############################################################################
            # YOUR CODE HERE
            # raise NotImplementedError()
            x_minus = inpainting(energy_model,x_minus,torch.zeros_like(x_minus),n_sample_steps,step_lr,langevin_grad_norm).detach()
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################

            # extend buffer
            if buffer_ptr + bs <= buffer_maxsize:
                replay_buffer[buffer_ptr: buffer_ptr +
                              bs] = ((x_minus * 255).to(torch.uint8).float() / 255).cpu()
            else:
                x_minus_ = (
                    (x_minus * 255).to(torch.uint8).float() / 255).cpu()
                replay_buffer[buffer_ptr:] = x_minus_[
                    :buffer_maxsize - buffer_ptr]
                remaining = bs - (buffer_maxsize - buffer_ptr)
                replay_buffer[:remaining] = x_minus_[
                    buffer_maxsize - buffer_ptr:]
            buffer_ptr = (buffer_ptr + bs) % buffer_maxsize
            buffer_size = min(buffer_maxsize, buffer_size + bs)

            # compute loss
            energy_model.train()
            x_plus = x_plus.to(device)
            x_minus = x_minus.to(device)
            e_plus = energy_model(x_plus)
            e_minus = energy_model(x_minus)
            ##############################################################################
            #                  TODO: You need to complete the code here                  #
            ##############################################################################
            # YOUR CODE HERE
            # raise NotImplementedError()
            optimizer.zero_grad()
            loss = ((l2_alpha * (e_plus ** 2 + e_minus ** 2) + (e_plus - e_minus))).sum()
            loss.backward()
            optimizer.step()
            ##############################################################################
            #                              END OF YOUR CODE                              #
            ##############################################################################

            train_loss += loss.sum().item()
            energy_plus += e_plus.sum().item()
            energy_minus += e_minus.sum().item()

            pbar.update(x_plus.size(0))
            pbar.set_description("Train Epoch {}, Train Loss: {:.6f}, ".format(epoch + 1, train_loss / n_batches) +
                                 "Energy Before Sampling: {:.6f}, ".format(energy_before / n_batches) +
                                 "Energy After Sampling: {:.6f}, ".format(energy_minus / n_batches) +
                                 "Energy of Ground Truth: {:.6f}".format(energy_plus / n_batches))
        pbar.close()

        if (epoch + 1) % save_interval == 0:
            os.makedirs(f'./ebm/{epoch + 1}', exist_ok=True)
            energy_model.eval()
            save_model(f'./ebm/{epoch + 1}/ebm.pth',
                       energy_model, optimizer, replay_buffer)

            # evaluate inpaiting
            # feel free to change the inpainting parameters!
            c_mse, r_mse, original, broken, recovered = evaluate(energy_model, val_loader,
                                                                N_SAMPLE_STEPS, STEP_LR, L_STEP_NORM, device=device)
            torchvision.utils.save_image(
                original, f"./ebm/{epoch + 1}/groundtruth.png", nrow=10)
            torchvision.utils.save_image(
                broken, f"./ebm/{epoch + 1}/corrupted.png", nrow=10)
            torchvision.utils.save_image(
                recovered, f"./ebm/{epoch + 1}/recovered.png", nrow=10)
            if r_mse < best_mse:
                print(f'Current best MSE: {best_mse} -> {r_mse}')
                best_mse = r_mse
                save_model('./ebm/ebm_best.pth', energy_model)
                
model = MlpBackbone((1, 28, 28), 1024).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

train_loader = DataLoader(train_set, 256, shuffle=True, drop_last=False, pin_memory=True)
val_loader = DataLoader(val_set, 500, shuffle=True, drop_last=False, pin_memory=True)

# feel free the change training hyper-parameters!
# train(20, model, train_loader, val_loader, opt imizer, 60, 1, 0.005, 0.03, 0.1)
# train(20, model, train_loader, val_loader, opt imizer, 60, 1, 0.005, 0.03, 0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))
train(20, model, train_loader, val_loader, optimizer,n_sample_steps=N_SAMPLE_STEPS,step_lr=STEP_LR,langevin_grad_norm=L_STEP_NORM,l2_alpha=0.3,langevin_eps=None)

# feel free to change evaluation parameters!
# inpainting parameters are not necessarily the same as sampling parameters
n_sample_steps = N_SAMPLE_STEPS
step_lr = STEP_LR
langevin_grad_norm = L_STEP_NORM

model.load_state_dict(load_model('./ebm/ebm_best.pth')[0])
corruption_mse, mse, _, _, _ = evaluate(model, val_loader, n_sample_steps, step_lr, langevin_grad_norm, device=device)
print(f'Corruption MSE: {corruption_mse}')
print(f'Recovered MSE: {mse}')


corruption_mse, mse, data, broken, revovered = evaluate(model, val_loader, n_sample_steps, step_lr, langevin_grad_norm, device=device)
print(f'Corruption MSE: {corruption_mse}')
print(f'Recovered MSE: {mse}')
torchvision.utils.save_image(data,'./ebm/data.png',nrow=10)
torchvision.utils.save_image(broken,'./ebm/data_corrupted.png',nrow=10)
torchvision.utils.save_image(revovered,'./ebm/data_recovered.png',nrow=10)