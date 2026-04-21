import typer 
import os
from dotenv import load_dotenv
from core.logging import logger

load_dotenv()
app = typer.Typer()

@app.command()
def train(
    model:  str = typer.Option("segformer", help="Model: [unet, unetpp, manet, segformer]"),
    encoder:str = typer.Option("mit_b2",    help="Encoder type"),    
    epochs: int = typer.Option(50,          help="Number of epochs"),
    device: str = typer.Option("auto",      help="Device: auto, cpu, cuda"),
):
    
    from training.config import ConfigurationManager
    from training.components import Training
    import mlflow
    
    # Setup MLFLOW
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_REPO_OWNER"]
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    logger.info(f"Training {model} + {encoder} for {epochs} epochs")
    
    config_manager = ConfigurationManager()
    config         = config_manager.get_training_config()
    
    # Override params with CLI args
    from dataclasses import replace
    config = replace(
        config,
        model = replace(config.model, model_name=model, encoder=encoder),
        train = replace(config.train, epochs=epochs),
    )
    
    training = Training(config=config)
    with mlflow.start_run(run_name=f"{model}_{encoder}"):
        training.train()

if __name__ == "__main__":
    app()