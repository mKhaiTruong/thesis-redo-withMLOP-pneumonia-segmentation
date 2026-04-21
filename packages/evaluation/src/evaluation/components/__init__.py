import sys, json, time, numpy as np, torch
import onnxruntime as ort
import segmentation_models_pytorch as smp

from core.logging import logger
from core.exception import CustomException
from evaluation.utils.data_loaders import get_eval_dataloader
from evaluation.utils.metrics.iou import compute_iou
from evaluation import EvaluationConfig

MODEL_MAP = {
    "unet":        smp.Unet,
    "unetpp":      smp.UnetPlusPlus,
    "deeplabv3":   smp.DeepLabV3,
    "segformer":   smp.Segformer,
    "manet":       smp.MAnet,
}

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config     = config
        self.val_loader = get_eval_dataloader(self.config)
        self.criterion  = self._get_loss_function()
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_loss_function(self):
        dice  = smp.losses.DiceLoss(mode='binary')
        focal = smp.losses.FocalLoss(mode='binary')
        return lambda preds, masks: dice(preds, masks) + focal(preds, masks)

    def _load_pytorch_model(self) -> torch.nn.Module:
        run_info = json.loads(
            (self.config.model.model_dir.parent / "run_info.json").read_text()
        )
        model = MODEL_MAP[run_info["model_name"]](
            encoder_name    = run_info["encoder"],
            encoder_weights = None,
            classes         = 1,
            activation      = None,
        ).to(self.device)

        model.load_state_dict(
            torch.load(self.config.model.model_dir, map_location=self.device)
        )
        model.eval()
        return model

    def _load_onnx_model(self, path: str, use_cuda: bool = False) -> ort.InferenceSession | None:
        if use_cuda:
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                logger.warning("CUDA not available — skipping CUDA ONNX")
                return None
            return ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    @torch.no_grad()
    def _eval_pytorch(self, model: torch.nn.Module) -> dict:
        total_loss, iou_scores = 0.0, []
        start = time.perf_counter()

        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            masks  = batch['mask'].to(self.device)
            outputs = model(images)
            total_loss  += self.criterion(outputs, masks).item()
            iou_scores.append(compute_iou(outputs, masks, threshold=self.config.eval.threshold))

        elapsed = (time.perf_counter() - start) * 1000 / len(self.val_loader)
        return {
            "loss":   round(total_loss / len(self.val_loader), 4),
            "iou":    round(float(np.mean(iou_scores)), 4),
            "avg_ms": round(elapsed, 2),
        }

    @torch.no_grad()
    def _eval_onnx(self, session: ort.InferenceSession) -> dict:
        total_loss, iou_scores = 0.0, []
        start = time.perf_counter()

        for batch in self.val_loader:
            images, masks = batch['image'], batch['mask']
            ort_inputs    = {session.get_inputs()[0].name: images.numpy()}
            outputs       = torch.tensor(session.run(None, ort_inputs)[0])
            total_loss   += self.criterion(outputs, masks).item()
            iou_scores.append(compute_iou(outputs, masks, threshold=self.config.eval.threshold))

        elapsed = (time.perf_counter() - start) * 1000 / len(self.val_loader)
        return {
            "loss":   round(total_loss / len(self.val_loader), 4),
            "iou":    round(float(np.mean(iou_scores)), 4),
            "avg_ms": round(elapsed, 2),
        }

    def validate(self) -> dict:
        try:
            results = {}

            # --- PyTorch ---
            logger.info("Evaluating PyTorch...")
            try:
                model, _ = self._load_pytorch_model()
                results["pytorch"] = self._eval_pytorch(model)
            except Exception as e:
                logger.warning(f"Skipping PyTorch: {e}")

            # --- ONNX ---
            onnx_variants = [
                ("onnx_fp32_cpu",  self.config.onnx.onnx_dir,      False),
                ("onnx_int8_cpu",  self.config.onnx.onnx_int8_dir, False),
                ("onnx_fp32_cuda", self.config.onnx.onnx_dir,      True),
                ("onnx_int8_cuda", self.config.onnx.onnx_int8_dir, True),
            ]
            for name, path, use_cuda in onnx_variants:
                if not path.exists():
                    logger.warning(f"Skipping {name}: not found")
                    continue
                session = self._load_onnx_model(path, use_cuda=use_cuda)
                if session:
                    logger.info(f"Evaluating {name}...")
                    results[name] = self._eval_onnx(session)

            self._print_table(results)
            self._save_metrics(results)
            return results

        except Exception as e:
            raise CustomException(e, sys)

    def _print_table(self, results: dict):
        print(f"\n{'Model':<30} {'IoU':>8} {'Loss':>8} {'ms/batch':>10}")
        print("-" * 58)
        for name, metrics in results.items():
            print(f"{name:<30} {metrics['iou']:>8.4f} {metrics['loss']:>8.4f} {metrics.get('avg_ms', 0):>10.2f}")

    def _save_metrics(self, results: dict):
        path = self.config.root_dir / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Metrics saved -> {path}")