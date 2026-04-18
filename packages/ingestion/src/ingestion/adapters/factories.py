from ingestion.adapters.local_ingestion_adapter import LocalIngestionAdapter
from ingestion.adapters.kaggle_ingestion_adapter import KaggleIngestionAdapter

class IngestionAdapterFactory:
    _adapters = {
        "LOCAL": LocalIngestionAdapter,
        "KAGGLE": KaggleIngestionAdapter
    }
    
    @staticmethod
    def create_adapter(source_type: str, source: str):
        source_type = source_type.upper()
        adapter_class = IngestionAdapterFactory._adapters.get(source_type)
        
        if not adapter_class:
            raise ValueError(f"Ingestion type '{source_type}' is not supported.")
        return adapter_class(source)