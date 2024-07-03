import torch
import tqdm
import gym
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential,
    TensorDictSequential as Seq,
)
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import ConvNet, EGreedyModule, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)

env = gym.make("MineRLObtainDiamondShovel-v0")

td = env.reset()