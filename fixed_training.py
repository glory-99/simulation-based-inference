import numpy as np
import numpy.random as rng
import torch
import torch.nn as nn
import time
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Generator(object):
    def __init__(self, p, theta, beta_range, N, X, cat_XtY=False) -> None:
        self.p = p
        self.theta = theta
        self.beta_range = beta_range
        self.N = N
        self.X = X
        self.cat_XtY = cat_XtY
        self.normalize_c = 1.

    def generate_samples(self, n):
        scale = self.beta_range[1] - self.beta_range[0]
        # every beta has the same theta
        theta = np.ones((n, self.p)) * self.theta
        gamma = rng.binomial(1, theta)
        beta = np.zeros((n, self.p))
        beta[gamma == 1] = rng.rand(
            np.sum(gamma == 1)) * scale + self.beta_range[0]
        beta[gamma == 0] = 0.
        Y = beta@self.X.T + rng.randn(n, self.N)
        if self.cat_XtY:
            Y = np.concatenate((Y, Y @ self.X), axis=1)
        return gamma, beta, Y/self.normalize_c


class MLP_pro(nn.Module):
    def __init__(self, N, p):
        super(MLP_pro, self).__init__()

        self.fc1 = nn.Linear(N, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, p)
        self.relu = nn.ReLU()
        self.mseloss = nn.MSELoss()
        self.bceloss = nn.BCEWithLogitsLoss()

    def forward(self, input):
        u = self.relu(self.fc1(input))
        u = self.relu(self.fc2(u))
        u = self.relu(self.fc3(u))
        u = self.relu(self.fc4(u))
        output = self.fc5(u)
        return output

    def get_mseloss(self, data, targ):
        output = self.forward(data)
        loss = self.mseloss(output, targ)
        return loss

    def get_bceloss(self, data, targ):
        output = self.forward(data)
        loss = self.bceloss(output, targ)
        return loss


def train_epoch(model, optimizer, train_data, train_labels, batch_size, bceloss=False, subset=0):
    n = train_data.shape[0]
    total_loss = 0.
    for i in range(math.ceil(n/batch_size)):
        data = torch.from_numpy(train_data[(i*batch_size):min((i+1)*batch_size, n-1)]).type(torch.float).to(device)
        targ = torch.from_numpy(train_labels[(i*batch_size):min((i+1)*batch_size, n-1)]).type(torch.float).to(device)
        if subset != 0:
            targ = targ[:, :subset]
        if bceloss:
            loss = model.get_bceloss(data, targ)
        else:
            loss = model.get_mseloss(data, targ)
        total_loss += loss.item() * data.shape[0]
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_loss/n

def train_model(model, lr, batch_size, epochs, train_data, train_labels, bceloss=False, subset=0, val_data=None, val_labels=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for i in range(epochs):
        model.train()
        train_loss = train_epoch(model, optimizer, train_data, train_labels, batch_size, bceloss, subset)
        train_losses.append(train_loss)
        print('Epoch: {}'.format(i+1))
        print('Train loss: {:.5f}'.format(train_loss))
        if isinstance(val_data, np.ndarray):
            val_loss = model_test(model, val_data, val_labels, bceloss)
            print('Val loss: {:.5f}'.format(val_loss))
            val_losses.append(val_loss)
    return train_losses, val_losses


def model_test(model, test_data, test_label, bceloss=False):
    model.eval()
    with torch.no_grad():
        data = torch.from_numpy(test_data).type(torch.float).to(device)
        targ = torch.from_numpy(test_label).type(torch.float).to(device)
        if bceloss:
            loss = model.get_bceloss(data, targ)
        else:
            loss = model.get_mseloss(data, targ)
    return loss.item()

if __name__ == "__main__":
    start_time = time.time()
    p = 200
    N = 50
    theta = 0.05
    beta_range = (-3, 3)
    X = np.load("./data/X_rho0_N50_p200.npy")
    normalize_c = np.load("./data/normalize_constant_rho0.npy")
    gamma_val = np.load('./data/gamma_val_rho0.npy')
    beta_val = np.load('./data/beta_val_rho0.npy')
    Y_val = np.load('./data/Y_val_rho0.npy') / normalize_c
    generator = Generator(p, theta, beta_range, N, X, cat_XtY=True)
    generator.normalize_c = normalize_c    #Pass the normalization constant to the generator
    gamma_train, beta_train, Y_train = generator.generate_samples(10000000)
    model = MLP_pro(N+p, p).to(device)
    print(device)
    train_losses, val_losses = train_model(model=model,
                                           lr=0.001,
                                           batch_size=256,
                                           epochs=200,
                                           train_data=Y_train,
                                           train_labels=beta_train,
                                           bceloss=False,
                                           subset=0,
                                           val_data=Y_val,
                                           val_labels=beta_val)
    end_time = time.time()
    torch.save(model.state_dict(), './model/model_rho0_fixedset.pt')
    np.save('./results/train_losses_rho0_fixedset', train_losses)
    np.save('./results/val_losses_rho0_fixedset', val_losses)
    print("Time: {:.2f}".format(end_time-start_time))


