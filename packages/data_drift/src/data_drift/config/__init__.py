from core.constants import *
from core.utils import read_yaml, create_directories
from data_drift import DataDriftMetricsConfig, DataDriftConfig

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_drift_config(self) -> DataDriftConfig:
        config = self.config.data_drift_config
        params = self.params.data_drift_params
        create_directories([config.root_dir])
        
        return DataDriftConfig(
            root_dir=Path(config.root_dir),
            origin_data_source = Path(self.config.data_transformation_config.out_train_dir) / "img",
            baseline_dir = Path(config.root_dir) / "baseline_distribution.npy",
            metric = DataDriftMetricsConfig(
                drift_threshold = params.drift_threshold,
                n_bins = params.n_bins,
                max_samples = params.max_samples,
                seed = params.seed,
                model_name = "resnet50.onnx",
            )
        )