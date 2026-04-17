from pneumonia_segmentation.adapters.ingestion.local_ingestion_adapter import LocalIngestionAdapter
from pneumonia_segmentation.adapters.ingestion.kaggle_ingestion_adapter import KaggleIngestionAdapter
from pneumonia_segmentation.adapters.transformation.covid_scan_adapter import CovidScanAdapter
from pneumonia_segmentation.adapters.transformation.lesion_segmenbtation_adapter import LesionSegmentationAdapter

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

class TransformationAdapterFactory:
    @staticmethod
    def get_adapter(name: str, path: str):
        name = name.upper()
        
        if name == "COVID-19_CT_SCANS":
            return CovidScanAdapter(path)
        elif name == "LESION_SEGMENTATION":
            return LesionSegmentationAdapter(path)
        else:
            raise ValueError(f"Source type '{name}' is not supported by AdapterFactory.")