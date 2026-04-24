from core.constants import *
from core.utils import read_yaml, create_directories

from inference import (
    DataDriftMetricsConfig, DataDriftConfig,
    ModelConfig, OnnxModelConfig, EvalDataConfig, EvaluationParamsConfig, 
    EvaluationConfig
)

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH, 
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def _model_slug(self) -> str:
        p = self.params.prepare_base_model_params
        return f"{p.model_name}_{p.encoder}"

    def _get_onnx_model_config(self, onnx_root: Path) -> OnnxModelConfig:
        slug = self._model_slug()
        d    = onnx_root / slug
        
        return OnnxModelConfig(
            onnx_dir        = d / "model.onnx",
            onnx_int8_dir   = d / "model_int8.onnx"
        )
        
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config.evaluation_config
        eval_params = self.params.evaluation_params
        create_directories([eval_config.root_dir])
        
        slug = self._model_slug()
        train_root  = Path(self.config.training_config.root_dir)
        onnx_root   = Path(self.config.onnx_config.root_dir)
        p = self.params.prepare_base_model_params

        evaluation_config = EvaluationConfig(
            root_dir = Path(eval_config.root_dir),
            model = ModelConfig(
                model_name  = p.model_name,
                encoder     = p.encoder,
                model_dir   = train_root / slug / "best_model.pth",
            ),
            onnx = OnnxModelConfig(
                onnx_dir        = onnx_root / slug / "model.onnx",
                onnx_int8_dir   = onnx_root / slug / "model_int8.onnx"
            ),
            data = EvalDataConfig(
                infer_data_dir = Path(
                    self.config.data_transformation_config.out_infer_dir
                ),
            ),
            eval = EvaluationParamsConfig(
                batch_size      = eval_params.batch_size,
                workers         = eval_params.workers,
                image_size      = eval_params.image_size,
                is_augmentation = eval_params.is_augmentation,
                threshold       = eval_params.threshold
            )
        )
        
        return evaluation_config
    
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
    
