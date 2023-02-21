import numpy as np
import numpy.random as rng
import torch 
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy 
from model import MLP_variant
from simulators import Generator_doubleNormal
import time
import argparse

def normalize_constant_est(generator, n=100000, seed=0):
    gamma_train, beta_train, Y_train = generator.generate_samples(n)
    mean = Y_train.mean(0)
    std = Y_train.std(0)
    return mean, std 

def model_test(model, data_loader, loss_type='mse', q=0.5, kwargs=None):
    model.eval()
    with torch.no_grad():
        n = 0 
        total_loss = 0.
        for _, (data, targ) in enumerate(data_loader):
            data, targ = data.to(device), targ.to(device)
            if kwargs:
                if 'subset' in kwargs:
                    targ = targ[:,(kwargs['subset'][0]-1):kwargs['subset'][1]]
            if loss_type == 'mse':
                loss = model.get_mseloss(data, targ)
            elif loss_type == 'bce':
                loss = model.get_bceloss(data, targ)
            elif loss_type == 'quantile':
                loss = model.get_quanloss(data, targ, q)
            elif loss_type == 'max_quantile':
                loss = model.get_max_quanloss(data, targ, q)
            total_loss += loss.item() * data.shape[0]
            n += data.shape[0]
    return total_loss/n

def predict(model, Y):
    model.eval()
    with torch.no_grad():
        data = torch.from_numpy(Y).type(torch.float).to(device)
        pred = model(data)
    return pred.detach().cpu().numpy()

'''
The following training function uses brand new samples at each batch.
'''
def train_epoch_with_generator(model, optimizer, generator, batch_size, iteration, loss_type, q, kwargs):
    model.train()
    train_loss = 0.
    for i in range(iteration):
        gamma, beta, Y = generator.generate_samples(batch_size)
        Y = (Y - kwargs['mean']) / kwargs['std']
        # print(np.sum(Y, 1)[0])

        if 'subset' in kwargs:
            gamma = torch.from_numpy(gamma[:,(kwargs['subset'][0]-1):kwargs['subset'][1]]).type(torch.float).to(device)
            beta = torch.from_numpy(beta[:,(kwargs['subset'][0]-1):kwargs['subset'][1]]).type(torch.float).to(device)
            Y = torch.from_numpy(Y).type(torch.float).to(device)
        else:
            gamma = torch.from_numpy(gamma).type(torch.float).to(device)
            beta = torch.from_numpy(beta).type(torch.float).to(device)
            Y = torch.from_numpy(Y).type(torch.float).to(device)

        if loss_type == 'mse':
            loss = model.get_mseloss(Y, beta)
        elif loss_type == 'bce':
            loss = model.get_bceloss(Y, gamma)
        elif loss_type == 'quantile':
            loss = model.get_quanloss(Y, beta, q)
        elif loss_type == 'max_quantile':
            loss = model.get_max_quanloss(Y, beta, q)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if 'scheduler' in kwargs:
        kwargs['scheduler'].step()
    return train_loss/(i+1)

# Input mean and std to normalize input.
# Input subset to take a subset of coordinates.
def train_model_with_generator(model, generator, optimizer, epochs, batch_size, iteration_per_epoch, loss_type='mse', q=0.5, val_data=None, **kwargs):
    assert loss_type in ['mse', 'bce', 'quantile', 'max_quantile']
    train_losses = []
    val_losses = []
    for i in range(epochs):
        train_loss = train_epoch_with_generator(
            model, optimizer, generator, batch_size, iteration_per_epoch, loss_type, q, kwargs)
        print('Epoch: {}'.format(i+1))
        print('Train loss: {:.5f}'.format(train_loss))
        train_losses.append(train_loss)
        if val_data.__str__() != 'None':
            val_loss = model_test(model, val_data, loss_type, q, kwargs)
            print('Val loss: {:.5f}'.format(val_loss))
            val_losses.append(val_loss)
        if 'model_list' in kwargs:
            if (i+1) in kwargs['save_point']:
                kwargs['model_list'].append(deepcopy(model.state_dict()))
        if 'coordinate_loss' in kwargs:
            pred = predict(model, kwargs['Y_test'])
            kwargs['coordinate_loss'].append(np.mean(np.maximum(q*(kwargs['beta_test']-pred),(1-q)*(pred-kwargs['beta_test'])), 0))
        if 'save_paras' in kwargs:
            for name, paras in model.named_parameters():
                if name in kwargs['save_paras']:
                    kwargs['paras_dict'][name].append(paras.detach().cpu().numpy())
    return train_losses, val_losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("p", type=int)
    parser.add_argument("q", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr_step_size", type=int)
    parser.add_argument("--exp_id", type=int)
    args = parser.parse_args()
    p = args.p
    theta = 0.05
    sigma0 = 0.1
    sigma1 = 5
    epochs = 1000
    init_lr = 0.001
    lr_step_size = 200 # lr scheduler step size
    lr_gamma = 0.4 # lr scheduler decreasing factor
    exp_id = 0
    q = args.q
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.epochs:
        epochs = args.epochs
    if args.lr_step_size:
        lr_step_size = args.lr_step_size
    if args.exp_id:
        exp_id = args.exp_id

    rng.seed(0)
    generator = Generator_doubleNormal(p, theta, sigma0, sigma1)
    mean, std = normalize_constant_est(generator)
    gamma_val, beta_val, Y_val = generator.generate_samples(1000000)
    val_dataset = TensorDataset(torch.Tensor((Y_val - mean) / std), torch.Tensor(beta_val))
    valid_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset))
    rng.seed()
    
    coordinate_loss = []
    start_time = time.time()
    model = MLP_variant(p, p, [1024, 1024], 'leakyrelu').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    train_losses, val_losses = train_model_with_generator(model, generator, optimizer, epochs=epochs,
                                        batch_size=256, iteration_per_epoch=4000, loss_type='quantile',
                                        q=q, val_data=valid_dataloader, scheduler=scheduler, mean=mean, std=std,
                                        coordinate_loss=coordinate_loss, Y_test=((Y_val - mean) / std), beta_test=beta_val)
    end_time = time.time()
    print("Trainging time: {:.2f}".format(end_time-start_time))
    np.save(f'./results/mean_p{p}_q{int(1000*q)}_exp{exp_id}', mean)
    np.save(f'./results/std_p{p}_q{int(1000*q)}_exp{exp_id}', std)
    np.save(f'./results/train_losses_p{p}_q{int(1000*q)}_exp{exp_id}', train_losses)
    np.save(f'./results/val_losses_p{p}_q{int(1000*q)}_exp{exp_id}', val_losses)
    np.save(f'./results/coordinate_loss_p{p}_q{int(1000*q)}_exp{exp_id}', coordinate_loss)
    torch.save(model.state_dict(), f'./model/p{p}_q{int(1000*q)}_exp{exp_id}.pt')
