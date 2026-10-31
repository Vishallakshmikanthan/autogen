"""Classical Machine Learning Engine (SYS-11, Module M12)."""

from autogen.intelligence.ml.applications import (
    AgentTrajectoryClusterer,
    CellDangerClassifier,
    LearnedPerceptionComponent,
    ResourceYieldPredictor,
)
from autogen.intelligence.ml.clustering import GaussianMixtureModel, KMeans
from autogen.intelligence.ml.contracts import (
    Classifier,
    Clusterer,
    DataLeakageError,
    Dataset,
    Estimator,
    ModelMetadata,
    Regressor,
    SplitResult,
    Transformer,
)
from autogen.intelligence.ml.data_pipeline import (
    GroupRunSplitter,
    MinMaxScaler,
    PolynomialFeatures,
    StandardScaler,
    check_data_leakage,
    group_k_fold,
)
from autogen.intelligence.ml.decomposition import PCA
from autogen.intelligence.ml.linear import LinearRegression, LogisticRegression
from autogen.intelligence.ml.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from autogen.intelligence.ml.neighbors import KNNClassifier, KNNRegressor
from autogen.intelligence.ml.serialization import load_linear_model, save_model_artifact
from autogen.intelligence.ml.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

__all__ = [
    "Classifier",
    "Clusterer",
    "DataLeakageError",
    "Dataset",
    "Estimator",
    "ModelMetadata",
    "Regressor",
    "SplitResult",
    "Transformer",
    "StandardScaler",
    "MinMaxScaler",
    "PolynomialFeatures",
    "GroupRunSplitter",
    "check_data_leakage",
    "group_k_fold",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
    "accuracy_score",
    "confusion_matrix",
    "precision_score",
    "recall_score",
    "f1_score",
    "log_loss",
    "roc_curve",
    "roc_auc_score",
    "LinearRegression",
    "LogisticRegression",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "KNNClassifier",
    "KNNRegressor",
    "KMeans",
    "GaussianMixtureModel",
    "PCA",
    "ResourceYieldPredictor",
    "CellDangerClassifier",
    "AgentTrajectoryClusterer",
    "LearnedPerceptionComponent",
    "save_model_artifact",
    "load_linear_model",
]

