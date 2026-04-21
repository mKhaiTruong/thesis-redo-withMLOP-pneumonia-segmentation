import typer 
import os
from dotenv import load_dotenv
from core.logging import logger

load_dotenv()
app = typer.Typer()

@app.command("validate")
def validate(
    model:  str = typer.Option("segformer", help="Model: [unet, unetpp, manet, segformer]"),
    encoder:str = typer.Option("mit_b2",    help="Encoder type"),
):
    
    from evaluation.config import ConfigurationManager
    from evaluation.components import Evaluation
    import mlflow
    
    # Setup MLFLOW
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_REPO_OWNER"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    logger.info(f"Evaluating {model} + {encoder}, both Pytorch and ONNX version")
    
    config_manager = ConfigurationManager()
    config         = config_manager.get_evaluation_config()
    
    # Override params with CLI args
    from dataclasses import replace
    config = replace(
        config,
        model = replace(config.model, model_name=model, encoder=encoder),
    )
    
    evaluation = Evaluation(config=config)
    with mlflow.start_run(run_name=f"{model}_{encoder}"):
        results = evaluation.validate()
        for model_type, metrics in results.items():
            mlflow.log_metrics({
                f"{model_type}_iou":    metrics["iou"],
                f"{model_type}_loss":   metrics["loss"],
                f"{model_type}_avg_ms": metrics["avg_ms"],
            })

if __name__ == "__main__":
    app()