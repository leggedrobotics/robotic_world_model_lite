from .base_cfg import BaseConfig
from dataclasses import dataclass, asdict, field
from typing import List, Dict


@dataclass
class AnymalDFlatConfig(BaseConfig):
    experiment_name: str = "offline"
    
    @dataclass
    class ExperimentConfig(BaseConfig.ExperimentConfig):
        environment: str = "anymal_d_flat"
    
    @dataclass
    class EnvironmentConfig(BaseConfig.EnvironmentConfig):
        reward_term_weights: Dict[str, float] = field(default_factory=lambda: {
            "track_lin_vel_xy_exp": 1.0,
            "track_ang_vel_z_exp": 0.5,
            "lin_vel_z_l2": -2.0,
            "ang_vel_xy_l2": -0.05,
            "dof_torques_l2": -2.5e-5,
            "dof_acc_l2": -2.5e-7,
            "action_rate_l2": -0.01,
            "feet_air_time": 0.5,
            "undesired_contacts": -1.0,
            "stand_still": -1.0,
            "flat_orientation_l2": -5.0,
            "dof_pos_limits": 0.0,
        })
        uncertainty_penalty_weight: float = -1.0
        command_resample_interval_range: List[int] | None = field(default_factory=lambda: [100, 120])
        event_interval_range: List[int] = field(default_factory=lambda: [48, 96])
    
    @dataclass
    class DataConfig(BaseConfig.DataConfig):
        dataset_root: str = "assets"
        dataset_folder: str = "data"
        batch_data_size: int = 10000
        state_idx_dict: Dict[str, List[int]] = field(default_factory=lambda: {
            r"$v$\n$[m/s]$": [0, 1, 2],
            r"$\omega$\n$[rad/s]$": [3, 4, 5],
            r"$g$\n$[1]$": [6, 7, 8],
            r"$q$\n$[rad]$": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            r"$\tau$\n$[Nm]$": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
        })
        state_data_mean: List[float] = field(default_factory=lambda: [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, -1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            -2.0, -2.0, 2.0, 2.0, -6.0, 8.0, -6.0, 8.0, 12.0, -12.0, 12.0, -12.0,
        ])
        state_data_std: List[float] = field(default_factory=lambda: [
            0.5, 0.5, 0.1,
            0.3, 0.3, 0.5,
            0.02, 0.02, 0.04,
            0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
            1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 2.5,
            15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
        ])
        action_data_mean: List[float] = field(default_factory=lambda: [0.0] * 12)
        action_data_std: List[float] = field(default_factory=lambda: [1.0] * 12)

    @dataclass
    class ModelArchitectureConfig(BaseConfig.ModelArchitectureConfig):
        history_horizon: int = 32 # the window size of the input state transitions
        forecast_horizon: int = 8 # the autoregressive prediction steps
        ensemble_size: int = 5
        contact_dim: int = 8
        termination_dim: int = 1
        architecture_config: Dict[str, object] = field(default_factory=lambda: {
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 256,
            "state_mean_shape": [128],
            "state_logstd_shape": [128],
            "extension_shape": [128],
            "contact_shape": [128],
            "termination_shape": [128],
        })
        resume_path: str | None = "assets/models/pretrain_rnn_ens.pt"
    
    @dataclass
    class PolicyArchitectureConfig(BaseConfig.PolicyArchitectureConfig):
        observation_dim: int = 48
        action_dim: int = 12
        resume_path: str | None = None

    @dataclass
    class PolicyAlgorithmConfig(BaseConfig.PolicyAlgorithmConfig):
        learning_rate: float = 1.0e-3
        entropy_coef: float = 0.005
    
    @dataclass
    class PolicyTrainingConfig(BaseConfig.PolicyTrainingConfig):
        save_interval: int = 50
        max_iterations: int = 500

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)