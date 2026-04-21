import typer 
from dotenv import load_dotenv
from core.logging import logger

load_dotenv()
app = typer.Typer()
        
@app.command("export")
def export(
    slug: str = typer.Option(..., help="Model slug, for example, segformer_mit_b2"),
):
    
    from onnx_export.config import ConfigurationManager
    from onnx_export.components import Onnx
    logger.info(f"Converting {slug} into ONNX format")
    
    config_manager = ConfigurationManager()
    config         = config_manager.get_onnx_config(slug=slug)
    onnx           = Onnx(config=config)
    onnx.export_onnx()
    onnx.quantize()

if __name__ == "__main__":
    app()