import os
from core.constants import *
from core.utils import read_yaml, create_directories
from training import (
    ModelConfig, DataConfig, OptimizerConfig, MetricConfig,
    TrainParamsConfig, TrainingConfig
)

class ConfigurationManager:
    def __init__(self, 
                 config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    # ── private helpers ───────────────────────────────────────
    def _model_slug(self) -> str:
        p = self.params.prepare_base_model_params
        return f"{p.model_name}_{p.encoder}"
    
    def _get_model_config(self, train_root: Path) -> ModelConfig:
        p    = self.params.prepare_base_model_params
        slug = self._model_slug()
        d    = train_root / slug
        os.makedirs(d, exist_ok=True)
        
        return ModelConfig(
            model_name      = p.model_name,
            encoder         = p.encoder,
            encoder_weights = p.encoder_weights,
            checkpoint_dir  = d / "checkpoints",
            latest_model_dir= d / "model.pth",
            best_model_dir  = d / "best_model.pth",
            run_info_dir    = d / "run_info.json"
            
        )
    
    def _get_data_config(self) -> DataConfig:
        c = self.config.data_transformation_config
        return DataConfig(
            train_data_dir  = Path(c.out_train_dir),
            valid_data_dir  = Path(c.out_valid_dir),  
        )
    
    def _get_optimizer_config(self) -> OptimizerConfig:
        p = self.params.training_params
        return OptimizerConfig(
            lr = p.lr,
            decay = p.decay,
            lr_scheduler = p.lr_scheduler
        )
    
    def _get_metric_config(self) -> MetricConfig:
        p = self.params.training_params
        return MetricConfig(
            metric_mode     = p.metric_mode,
            loss_function   = p.loss_function,
            alpha = p.alpha,
            gamma = p.gamma
        )
    
    def _get_train_params_config(self) -> TrainParamsConfig:
        p = self.params.training_params
        return TrainParamsConfig(
            batch_size      = p.batch_size,
            patience        = p.patience,
            start_epoch     = p.start_epoch,
            epochs          = p.epochs,
            workers         = p.workers,
            seed            = p.seed,
            image_size      = p.image_size,
            is_augmentation = p.is_augmentation,
        )

    # ── public ───────────────────────────────────────────────
    def get_training_config(self) -> TrainingConfig:
        train_root = Path(self.config.training_config.root_dir)
        create_directories([train_root])

        return TrainingConfig(
            root_dir  = train_root,
            model     = self._get_model_config(train_root),
            data      = self._get_data_config(),
            metric    = self._get_metric_config(),
            optimizer = self._get_optimizer_config(),
            train     = self._get_train_params_config(),
        )