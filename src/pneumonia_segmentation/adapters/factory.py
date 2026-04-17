from pneumonia_segmentation.adapters.local_ingestion_adapter import LocalIngestionAdapter
from pneumonia_segmentation.adapters.kaggle_ingestion_adapter import KaggleIngestionAdapter
from pneumonia_segmentation.adapters.nifti_adapter import NiftiAdapter

class IngestionAdapterFactory:
    _adapters = {
        "LOCAL": LocalIngestionAdapter,
        "KAGGLE": KaggleIngestionAdapter
    }
    
    @staticmethod
    def create_adapter(ingestion_type: str, source: str):
        ingestion_type = ingestion_type.upper()
        adapter_class = IngestionAdapterFactory._adapters.get(ingestion_type)
        
        if not adapter_class:
            raise ValueError(f"Ingestion type '{ingestion_type}' is not supported.")
        return adapter_class(source)

class TransformationAdapterFactory:
    @staticmethod
    def get_adapter(source_type: str, path: str):
        source_type = source_type.upper()
        
        if source_type == "NIFTI":
            return NiftiAdapter(path)
        else:
            raise ValueError(f"Source type '{source_type}' is not supported by AdapterFactory.")