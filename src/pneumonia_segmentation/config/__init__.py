import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pneumonia_segmentation.constants import *
from pneumonia_segmentation.utils.common import read_yaml, create_directories

from pneumonia_segmentation.entity.entity_config import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataDriftMetricsConfig, DataDriftConfig,
    ModelArchitecture, PrepareBaseModelConfig,
    ModelConfig, DataConfig, OptimizerConfig, MetricConfig, TrainParamsConfig, 
    TrainingConfig,
    TrainedModelConfig, OnnxConfig,
    TensorRTEngineConfig, EvaluationConfig, EvalDataConfig, EvaluationParamsConfig,
    OnnxModelConfig, TensorRTConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion_config
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir    = config.root_dir,
            source_type = os.getenv("DATA_SOURCE_TYPE", "LOCAL"),
            source      = os.getenv("DATA_SOURCE"),
            name        = os.getenv("DATA_SOURCE_NAME"),
        )

        return data_ingestion_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation_config
        params = self.params.data_transformation_params

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir               = config.root_dir,
            source_type            = config.source_type,
            data_dir               = config.data_dir,
            out_train_dir          = config.out_train_dir,
            out_valid_dir          = config.out_valid_dir,
            out_infer_dir          = config.out_infer_dir,
            params_image_size      = params.image_size,
            params_skip_background_ratio = params.skip_background_ratio,
            params_slice_interval  = params.slice_interval,
            params_valid_size      = params.valid_size,
            params_infer_size      = params.infer_size
        )

        return data_transformation_config

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
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model_config
        params = self.params.prepare_base_model_params
        
        model_dir = os.path.join(config.root_dir, f"{params.model_name}_{params.encoder}")
        create_directories([config.root_dir, model_dir])
        
        modelArchitecture = ModelArchitecture(
            model_architecture = params.model_architecture,
            library = params.library,
            model_name = params.model_name,
            encoder = params.encoder,
            encoder_weights = params.encoder_weights,
            classes = params.classes,
            activation = params.activation,
        )

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(os.path.join(model_dir, "base_model.pth")),
            modelArchitecture=modelArchitecture
        )
        
        return prepare_base_model_config

    # ----------- TRAINING -----------------
    
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
    
    # --------------------- ONNX ---------------------------
    
    # ── private helpers ───────────────────────────────────────

    def _get_trained_model_config(self, train_root: Path) -> TrainedModelConfig:
        slug = self._model_slug()
        d    = train_root / slug
        
        return TrainedModelConfig(
            checkpoint_dir  = d / "checkpoints",
            latest_model_dir= d / "model.pth",
            best_model_dir  = d / "best_model.pth",
            run_info_dir    = d / "run_info.json"
        )
        
    # ── public ───────────────────────────────────────────────
    def get_onnx_config(self) -> OnnxConfig:
        onnx_root  = Path(self.config.onnx_config.root_dir) 
        train_root = Path(self.config.training_config.root_dir)
        slug       = self._model_slug()

        return OnnxConfig(
            root_dir        = onnx_root,
            trained_model   = self._get_trained_model_config(train_root),
            onnx_model_dir      = onnx_root / slug / "model.onnx",
            onnx_int8_model_dir = onnx_root / slug / "model_int8.onnx",
            image_size      = self.params.onnx_params.image_size
        )
    
    # --------------------- Evaluation ---------------------------
    
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
        onnx_root   = Path(self.config.onnx_config.root_dir)
        engine_root = Path(self.config.tensorRT_config.root_dir)

        return EvaluationConfig(
            root_dir = Path(eval_config.root_dir),
            onnx = OnnxModelConfig(
                onnx_dir        = onnx_root / slug / "model.onnx",
                onnx_int8_dir   = onnx_root / slug / "model_int8.onnx"
            ),
            engine = TensorRTEngineConfig(
                engine_dir = engine_root / slug / "model.engine",
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
    
    # --------------------- TensorRT ---------------------------
    def get_tensorrt_config(self) -> TensorRTConfig:
        config = self.config.tensorRT_config
        params = self.params.tensorRT_params
        create_directories([config.root_dir])
        
        onnx_config = self.config.onnx_config
        root_dir = Path(config.root_dir)
        slug = self._model_slug()
        
        return TensorRTConfig(
            root_dir    = root_dir,
            out_dir     = root_dir / slug / "model.engine",
            image_size  = params.image_size,
            onnx = OnnxModelConfig(
                onnx_dir      = Path(onnx_config.root_dir) / slug / "model.onnx",
                onnx_int8_dir = Path(onnx_config.root_dir) / slug / "model_int8.onnx"
            )
        )
    
