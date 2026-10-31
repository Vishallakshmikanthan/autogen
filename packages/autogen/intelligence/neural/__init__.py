"""Neural network and deep reinforcement learning engine (SYS-13, Module M14)."""

from autogen.intelligence.neural.activations import (
    Activation,
    LeakyReLU,
    Linear,
    ReLU,
    Sigmoid,
    Softmax,
    Tanh,
)
from autogen.intelligence.neural.cnn import CNNTerrainClassifier
from autogen.intelligence.neural.contracts import (
    ExperienceTransition,
    Layer,
    LossFunction,
    NetworkDiagnostics,
    Optimizer,
)
from autogen.intelligence.neural.dqn import DeepQNetwork, ReplayBuffer
from autogen.intelligence.neural.layers import Conv2DLayer, DenseLayer, DropoutLayer, FlattenLayer
from autogen.intelligence.neural.losses import (
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HuberLoss,
    MeanSquaredError,
)
from autogen.intelligence.neural.mlp import NeuralNetwork
from autogen.intelligence.neural.optimizers import (
    SGD,
    Adam,
    CosineAnnealingLR,
    ExponentialLR,
    Momentum,
    RMSProp,
    StepLR,
    clip_grad_norm,
    clip_grad_value,
)
from autogen.intelligence.neural.policy_gradient import PPO, REINFORCE
from autogen.intelligence.neural.survival_predictor import NeuralSurvivalPredictor
from autogen.intelligence.neural.torch_twin import PyTorchTwin, is_torch_available

__all__ = [
    "Activation",
    "Sigmoid",
    "Tanh",
    "ReLU",
    "LeakyReLU",
    "Linear",
    "Softmax",
    "LossFunction",
    "MeanSquaredError",
    "BinaryCrossEntropy",
    "CategoricalCrossEntropy",
    "HuberLoss",
    "Layer",
    "DenseLayer",
    "DropoutLayer",
    "FlattenLayer",
    "Conv2DLayer",
    "Optimizer",
    "SGD",
    "Momentum",
    "RMSProp",
    "Adam",
    "StepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "clip_grad_norm",
    "clip_grad_value",
    "NeuralNetwork",
    "CNNTerrainClassifier",
    "PyTorchTwin",
    "is_torch_available",
    "ReplayBuffer",
    "DeepQNetwork",
    "ExperienceTransition",
    "NetworkDiagnostics",
    "REINFORCE",
    "PPO",
    "NeuralSurvivalPredictor",
]

