import sys, json, time, numpy as np, torch
import onnxruntime as ort
import segmentation_models_pytorch as smp

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException
from pneumonia_segmentation.utils.data_loaders import get_eval_dataloader
from pneumonia_segmentation.utils.metrics.iou import compute_iou
from pneumonia_segmentation.entity.entity_config import EvaluationConfig

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config       = config
        self.val_loader   = get_eval_dataloader(self.config)
        self.criterion    = self._get_loss_function()
    
    def _load_cpu_onnx_model(self, path: str) -> ort.InferenceSession:
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    
    def _load_cuda_onnx_model(self, path: str) -> ort.InferenceSession:
        return ort.InferenceSession(str(path), providers=["CUDAExecutionProvider"])
    
    def _get_loss_function(self):
        dice = smp.losses.DiceLoss(mode='binary')
        focal = smp.losses.FocalLoss(mode='binary')
        return lambda preds, masks: dice(preds, masks) + focal(preds, masks)
    
    @torch.no_grad()
    def _eval_onnx(self, session: ort.InferenceSession) -> dict:
        start = time.perf_counter()
        total_loss, iou_scores = 0.0, []
        
        for batch in self.val_loader:
            images, masks = batch['image'], batch['mask']
            ort_inputs    = {session.get_inputs()[0].name: images.numpy()}
            outputs       = torch.tensor(session.run(None, ort_inputs)[0])
            total_loss   += self.criterion(outputs, masks).item()
            iou_scores.append(
                compute_iou(outputs, masks, threshold=self.config.eval.threshold))
        
        elapsed = (time.perf_counter() - start) * 1000 / len(self.val_loader)
        return {
            "loss": round(total_loss / len(self.val_loader), 4),
            "iou":  round(float(np.mean(iou_scores)), 4),
            "avg_ms":  round(elapsed, 2),
        }
    
    @torch.no_grad()
    def _eval_tensorrt(self) -> dict | None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            logger  = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)

            with open(self.config.engine.engine_dir, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())

            context = engine.create_execution_context()
            start   = time.perf_counter()
            total_loss, iou_scores = 0.0, []

            for batch in self.val_loader:
                images, masks = batch['image'], batch['mask']
                images_np     = images.numpy().astype(np.float32)

                # Set input shape first, then query output shape
                context.set_input_shape("input", images_np.shape)
                output_shape = tuple(context.get_tensor_shape("output"))
                output       = np.empty(output_shape, dtype=np.float32)

                input_mem  = cuda.mem_alloc(images_np.nbytes)
                output_mem = cuda.mem_alloc(output.nbytes)

                cuda.memcpy_htod(input_mem, images_np)
                context.execute_v2([int(input_mem), int(output_mem)])
                cuda.memcpy_dtoh(output, output_mem)

                outputs     = torch.tensor(output)
                total_loss += self.criterion(outputs, masks).item()
                iou_scores.append(
                    compute_iou(outputs, masks, threshold=self.config.eval.threshold))
            
            elapsed = (time.perf_counter() - start) * 1000 / len(self.val_loader)
            return {
                "loss":   round(total_loss / len(self.val_loader), 4),
                "iou":    round(float(np.mean(iou_scores)), 4),
                "avg_ms": round(elapsed, 2),
            }

        except ImportError:
            logging.info("TensorRT not available, skipping...")
            return None
        
    @torch.no_grad()
    def validate(self):
        try:
            results = {}
            
            logging.info("Evaluating CPU ONNX FP32...")
            results["cpu_onnx_fp32"] = self._eval_onnx(
                self._load_cpu_onnx_model(str(self.config.onnx.onnx_dir)))
            
            logging.info("Evaluating CUDA ONNX FP32...")
            results["cuda_onnx_fp32"] = self._eval_onnx(
                self._load_cuda_onnx_model(str(self.config.onnx.onnx_dir)))
            
            logging.info("Evaluating CPU ONNX INT8...")
            results["cpu_onnx_int8"] = self._eval_onnx(
                self._load_cpu_onnx_model(str(self.config.onnx.onnx_int8_dir)))
            
            logging.info("Evaluating CUDA ONNX INT8...")
            results["cuda_onnx_int8"] = self._eval_onnx(
                self._load_cuda_onnx_model(str(self.config.onnx.onnx_int8_dir)))
            
            logging.info("Evaluating TensorRT...")
            trt_result = self._eval_tensorrt()
            if trt_result:
                results["tensorrt"] = trt_result
            
            self._print_table(results)
            self._save_metrics(results)
            return results
        
        except Exception as e:
            raise CustomException(e, sys)

    def _print_table(self, results: dict):
        print(f"\n{'Model':<20} {'IoU':>8} {'Loss':>8} {'ms/batch':>10}")
        print("-" * 48)
        for name, metrics in results.items():
            print(f"{name:<20} {metrics['iou']:>8.4f} {metrics['loss']:>8.4f} {metrics.get('avg_ms', 0):>10.2f}")

    def _save_metrics(self, results: dict):
        path = self.config.root_dir / self.config.onnx.onnx_int8_dir.parent.name / "metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Metrics saved -> {path} | {results   }")