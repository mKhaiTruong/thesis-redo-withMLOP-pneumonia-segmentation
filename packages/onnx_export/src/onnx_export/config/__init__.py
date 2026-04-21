from core.constants import *
from core.utils import read_yaml, create_directories
from onnx_export import (
    OnnxConfig,
    TrainedModelConfig
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
    def get_onnx_config(self, slug: str = None) -> OnnxConfig:
        onnx_root  = Path(self.config.onnx_config.root_dir) 
        train_root = Path(self.config.training_config.root_dir)
        slug       = self._model_slug() if slug is None else slug

        return OnnxConfig(
            root_dir        = onnx_root,
            trained_model   = self._get_trained_model_config(train_root),
            onnx_model_dir      = onnx_root / slug / "model.onnx",
            onnx_int8_model_dir = onnx_root / slug / "model_int8.onnx",
            image_size      = self.params.onnx_params.image_size
        )