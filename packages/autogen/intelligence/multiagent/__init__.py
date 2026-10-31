"""Multi-Agent Systems & Collective Intelligence package (SYS-14, Module M15)."""

from autogen.intelligence.multiagent.centralized_critic import CentralizedCriticMARL
from autogen.intelligence.multiagent.commons import CommonsResourcePool
from autogen.intelligence.multiagent.communication import CommunicationBus
from autogen.intelligence.multiagent.contracts import (
    JointTransition,
    MARLAlgorithm,
    Message,
    MultiAgentMetricSummary,
    RewardMode,
)
from autogen.intelligence.multiagent.coordination import (
    ContractAward,
    ContractNetProtocol,
    TaskAnnouncement,
    TaskBid,
    VickreyAuction,
)
from autogen.intelligence.multiagent.emergent_roles import RoleDetector
from autogen.intelligence.multiagent.iql import IndependentQLearning
from autogen.intelligence.multiagent.matrix_games import MatrixGame
from autogen.intelligence.multiagent.metrics import MultiAgentMetrics
from autogen.intelligence.multiagent.self_play import OpponentPool, PolicySnapshot
from autogen.intelligence.multiagent.stigmergy import PheromoneField
from autogen.intelligence.multiagent.value_decomposition import (
    QMIXLite,
    ValueDecompositionNetworks,
)

__all__ = [
    "RewardMode",
    "Message",
    "JointTransition",
    "MultiAgentMetricSummary",
    "MARLAlgorithm",
    "CommonsResourcePool",
    "CommunicationBus",
    "PheromoneField",
    "ContractAnnouncement",
    "TaskAnnouncement",
    "TaskBid",
    "ContractAward",
    "ContractNetProtocol",
    "VickreyAuction",
    "MatrixGame",
    "IndependentQLearning",
    "CentralizedCriticMARL",
    "ValueDecompositionNetworks",
    "QMIXLite",
    "OpponentPool",
    "PolicySnapshot",
    "MultiAgentMetrics",
    "RoleDetector",
]
