#!/usr/bin/env python
# coding: utf-8

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
from tqdm.rich import trange

from warnings import filterwarnings
import yaml
import os

filterwarnings("ignore")

CONFIG_DIR = "configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

CONFIGS = [
    "default.yaml",
    "high_dim.yaml",
    "low_corr.yaml",
]

def load_yaml_config(path: str):
    """
    Load a YAML configuration file and convert it to an object.

    Args:
        path (str): The path to the YAML configuration file.

    Returns:
        Config: An object representing the YAML configuration.

    """
    with open(path) as file:
        config = yaml.safe_load(file)
    # Convert to object
    config = type("Config", (object,), config)()
    # Convert numeric values to integers and floats
    for key, value in config.__dict__.items():
        if isinstance(value, (int, float)):
            setattr(config, key, type(value)(value))
    return config


# Load the config

"""Default configuration:
N: 10000                    # Number of samples
B: 1                        # Coefficient for the covariance matrix
r: 1                        # Coefficient for the covariance matrix
L: 3                        # Number of hidden layers
w: 128                      # Width of each hidden layer
p: 256                      # Number of features
noise: 0.1                  # Noise level
s: 0.1                      # Sparsity level
lr: 3e-4                    # Learning rate
batch_size: 512             # Batch size
n_bootstrap: 100            # Number of bootstrap samples
log_path: "logs/default"    # Path to save logs
"""


# MLP for generating data

class MLP(nn.Module):
    def __init__(self, L, w, p):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(p, w)])
        for _ in range(L):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(w, w))
        self.out = nn.Linear(w, 1)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)
        

# MLPRegressor, a PyTorch Lightning module for training the MLP
class MLPRegressor(pl.LightningModule):
    """
    Multi-Layer Perceptron (MLP) Regressor model implemented using PyTorch Lightning.

    Args:
        L (int): Number of hidden layers in the MLP.
        w (int): Width of each hidden layer in the MLP.
        p (float): Dropout probability for the hidden layers.
        lr (float): Learning rate for the optimizer.

    Attributes:
        model (MLP): The MLP model.
        lr (float): The learning rate for the optimizer.

    Methods:
        forward(x): Performs a forward pass through the MLP model.
        training_step(batch): Computes the loss and logs it during training.
        validation_step(batch): Computes the loss and logs it during validation.
        backward(loss): Performs backpropagation to compute gradients.
        configure_optimizers(): Configures the optimizer and learning rate scheduler.
        train_dataloader(X_train, y_train, batch_size): Returns a data loader for training data.
        val_dataloader(X_val, y_val, batch_size): Returns a data loader for validation data.
    """

    def __init__(self, L, w, p, lr):
        super().__init__()
        self.model = MLP(L, w, p)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        y_hat = self.model(x).squeeze()
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def backward(self, loss):
        loss.backward(retain_graph=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }

    def train_dataloader(self, X_train, y_train, batch_size):
        return DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True)
    
    def val_dataloader(self, X_val, y_val, batch_size):
        return DataLoader(list(zip(X_val, y_val)), batch_size=batch_size)



# Weight initialization kaiming normal
def init_weights(m):
    """
    Initialize the weights of a linear layer using the Kaiming normal initialization.

    Args:
        m (nn.Linear): The linear layer to initialize.

    Returns:
        None
    """
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)

# True MLP, only init




def generate_data(config: dict, MLP: nn.Module):
    """
    Generate synthetic data for training and validation.

    Args:
        config (dict): A dictionary containing configuration parameters.
        MLP (nn.Module, optional): The MLP model used for generating the labels. Defaults to MLP_true.

    Returns:
        tuple: A tuple containing the training and validation datasets for X and y.
    """
    
    # \Sigma_{ij} = r^\lvert i - j\rvert
    sigma = torch.tensor([[config.r**abs(i-j) for i in range(config.p)] for j in range(config.p)])
    # Non-zero covariates
    num_nonzero = int(config.s*config.p)
    # From multivariate normal distribution
    X = torch.distributions.MultivariateNormal(
        loc=torch.zeros(config.p), covariance_matrix=config.B*sigma
    ).sample((config.N,)).unsqueeze(1)
    X[:, :, num_nonzero:].zero_()
    # y = f^\ast(X) + \epsilon
    y = MLP(X).squeeze() + config.noise*torch.randn(config.N)
    X_train, X_val = random_split(X, [int(0.8*config.N), int(0.2*config.N)])
    y_train, y_val = random_split(y, [int(0.8*config.N), int(0.2*config.N)])
    return X_train.dataset, X_val.dataset, y_train.dataset, y_val.dataset


def perturb_data(X, y, level=0.2, M=0):
    """
    Perturbs the input data by masking a portion of the features for each sample.

    Args:
        X (torch.Tensor): The input data tensor.
        y (torch.Tensor): The target data tensor.
        level (float, optional): The probability of masking a feature. Defaults to 0.2.
        M (float, optional): The value to replace the masked features with. Defaults to 0.

    Returns:
        torch.Tensor: The perturbed input data tensor.
        torch.Tensor: The target data tensor.
    """
    mask = torch.rand(X.shape) < level
    X_perturbed = X.clone()
    X_perturbed[mask] = M
    return X_perturbed, y


# bootstrapping dataset
def bootstrapping(X, y):
    """
    Perform bootstrapping on the given data.

    Parameters:
    - X (torch.Tensor): The input data.
    - y (torch.Tensor): The target labels.

    Returns:
    - X_bootstrapped (torch.Tensor): The bootstrapped input data.
    - y_bootstrapped (torch.Tensor): The bootstrapped target labels.
    """
    indices = torch.randint(0, len(X), (len(X),))
    X_bootstrapped = X[indices]
    y_bootstrapped = y[indices]
    return X_bootstrapped, y_bootstrapped


class DataModule(pl.LightningDataModule):
    """
    LightningDataModule subclass for handling data loading and processing.

    Args:
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
    """

    def __init__(self, train_loader, val_loader):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_dataloader(self):
        """
        Returns the DataLoader for training data.
        """
        return self.train_loader

    def val_dataloader(self):
        """
        Returns the DataLoader for validation data.
        """
        return self.val_loader

for config_path in CONFIGS:
    config = load_yaml_config(os.path.join(CONFIG_DIR, config_path))
    print(config.__dict__)
    MLP_true = MLP(config.L, config.w, config.p)
    MLP_true.apply(init_weights)    # Initialize weights

    X_train, X_val, y_train, y_val = generate_data(config, MLP=MLP_true)

    for _ in trange(config.n_bootstrap):
        # Perturb data
        X_train_perturbed, y_train_perturbed = perturb_data(X_train, y_train)
        # Bootstrap samples
        X_train_perturbed, y_train_perturbed = bootstrapping(X_train_perturbed, y_train_perturbed)

        X_val_perturbed, y_val_perturbed = perturb_data(X_val, y_val)

        model = MLPRegressor(config.L, config.w, config.p, config.lr)
        model_perturbed = MLPRegressor(config.L, config.w, config.p, config.lr)

        # Dataloaders
        train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=config.batch_size, shuffle=True, pin_memory=True)
        train_loader_perturbed = DataLoader(list(zip(X_train_perturbed, y_train_perturbed)), batch_size=config.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(list(zip(X_val, y_val)), batch_size=config.batch_size, shuffle=False, pin_memory=True)

        # DataModules
        train_datamodel = DataModule(train_loader, val_loader)
        train_datamodel_perturbed = DataModule(train_loader_perturbed, val_loader)

        # Loggers
        logger = TensorBoardLogger(config.log_path, name="original")
        logger_perturbed = TensorBoardLogger(config.log_path, name="perturbed")

        # Trainers
        trainer = pl.Trainer(max_epochs=100, logger=logger)
        trainer_perturbed = pl.Trainer(max_epochs=100, logger=logger_perturbed)
        
        # Train
        trainer.fit(model, train_datamodel)
        trainer_perturbed.fit(model_perturbed, train_datamodel_perturbed)