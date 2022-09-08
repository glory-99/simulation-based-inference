import numpy as np
import numpy.random as rng
import torch
import torch.nn as nn
import time
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


def train_epoch_with_generator(model, optimizer, generator, batch_size, iteration, bceloss=False, subset=0):
    train_loss = 0.
    for i in range(iteration):
        gamma, beta, Y = generator.generate_samples(batch_size)
        gamma = torch.from_numpy(gamma).type(torch.float).to(device)
        beta = torch.from_numpy(beta).type(torch.float).to(device)
        Y = torch.from_numpy(Y).type(torch.float).to(device)
        if subset != 0:
            beta = beta[:, :subset]
            gamma = gamma[:, :subset]
        if bceloss:
            loss = model.get_bceloss(Y, gamma)
        else:
            loss = model.get_mseloss(Y, beta)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss/(i+1)


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


def train_model_with_generator(model, generator, lr, batch_size, epochs, iter_per_epoch, bceloss=False, subset=0, val_data=None, val_label=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    for i in range(epochs):
        model.train()
        train_loss = train_epoch_with_generator(
            model, optimizer, generator, batch_size, iter_per_epoch, bceloss, subset)
        print('Epoch: {}, Loss: {:.5f}'.format(i+1, train_loss))
        train_losses.append(train_loss)
        if isinstance(val_data, np.ndarray):
            val_loss = model_test(model, val_data, val_label, bceloss)
            print('Val loss: {:.5f}'.format(val_loss))
            val_losses.append(val_loss)
    return train_losses, val_losses


if __name__ == "__main__":
    start_time = time.time()
    p = 200
    N = 50
    theta = 0.05
    beta_range = (-3, 3)
    X = np.load("./data/X_rho0_N50_p200.npy")
    normalize_c = np.load("normalize_constant.npy")
    generator = Generator(p, theta, beta_range, N, X, cat_XtY=True)
    generator.normalize_c = normalize_c    #Pass the normalization constant to the generator
    gamma_val, beta_val, Y_val = generator.generate_samples(1000)
    model = MLP_pro(N+p, p).to(device)
    print(device)
    train_losses, val_losses = train_model_with_generator(model=model,
                                                          generator=generator,
                                                          lr=0.001,
                                                          batch_size=256,
                                                          epochs=1000,
                                                          iter_per_epoch=3000,
                                                          val_data=Y_val,
                                                          val_label=beta_val)
    end_time = time.time()
    torch.save(model.state_dict(), 'My_model_promax.pt')
    np.save('train_losses_promax', train_losses)
    np.save('val_losses_promax', val_losses)
    print("Time: {:.2f}".format(end_time-start_time))


