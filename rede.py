import numpy as np
import math

import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader
import torch.distributions.uniform as urand

import matplotlib.pyplot as plt


class Perceptron(object):

    def __init__(self, no_of_inputs):
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = summation
        else:
            activation = 0
        return activation

class AlgebraicDataset(Dataset):
    def __init__(self, f, interval, nsamples):
        X = urand.Uniform(interval[0], interval[1]).sample([nsamples])
        self.data = [(x, f(x)) for x in X]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

f = lambda x: math.cos(x/2)
interval = (-10, 10)
train_nsamples = 1000
test_nsamples = 100

train_dataset = AlgebraicDataset(f, interval, train_nsamples)
test_dataset = AlgebraicDataset(f, interval, test_nsamples)

train_dataloader = DataLoader(train_dataset, train_nsamples, shuffle=True)
test_dataloader = DataLoader(test_dataset, test_nsamples, shuffle=True)

#Arquitetura da Rede
class MultiLayerNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
        nn.Linear(1, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )

  def forward(self, x):
    return self.layers(x)

def train(model, dataloader, lossfunc, optimizer):
    model.train()
    cumloss = 0.0
    for X, y in dataloader:
        X = X.unsqueeze(1).float().to(device)
        y = y.unsqueeze(1).float().to(device)

        pred = model(X)
        loss = lossfunc(pred, y)

        # zera os gradientes acumulados
        optimizer.zero_grad()
        # computa os gradientes
        loss.backward()
        # anda, de fato, na direção que reduz o erro local
        optimizer.step()

        # loss é um tensor; item pra obter o float
        cumloss += loss.item()

    return cumloss / len(dataloader)


def test(model, dataloader, lossfunc):
    model.eval()

    cumloss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X = X.unsqueeze(1).float().to(device)
            y = y.unsqueeze(1).float().to(device)

            pred = model(X)
            loss = lossfunc(pred, y)
            cumloss += loss.item()

    return cumloss / len(dataloader)

# Pra visualizar
def plot_comparinson(f, model, interval=(-10, 10), nsamples=10):
  fig, ax = plt.subplots(figsize=(10, 10))

  ax.grid(True, which='both')
  ax.spines['left'].set_position('zero')
  ax.spines['right'].set_color('none')
  ax.spines['bottom'].set_position('zero')
  ax.spines['top'].set_color('none')

  samples = np.linspace(interval[0], interval[1], nsamples)
  model.eval()
  with torch.no_grad():
    pred = model(torch.tensor(samples).unsqueeze(1).float().to(device))

  ax.plot(samples, list(map(f, samples)), "o", label="ground truth")
  ax.plot(samples, pred.cpu(), label="model")
  plt.legend()
  plt.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Rodando na {device}")

multimodel = MultiLayerNetwork().to(device)

# Função de perda (loss function)
# Erro quadrático médio (Mean Squared Error)
lossfunc = nn.MSELoss()
# Gradiente Descendente Estocástico
# SGD = Stochastic Gradient Descent
optimizer = torch.optim.SGD(multimodel.parameters(), lr=1e-3)
# taxa de aprendizado lr = learning rate

epochs = 20001
for t in range(epochs):
  train_loss = train(multimodel, train_dataloader, lossfunc, optimizer)

plot_comparinson(f, multimodel, nsamples=40)
test_loss = test(multimodel, test_dataloader, lossfunc)
print(f"Test Loss: {test_loss}")
