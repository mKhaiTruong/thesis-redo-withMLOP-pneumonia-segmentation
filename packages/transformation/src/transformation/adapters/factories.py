from transformation.adapters.covid_scan_adapter import CovidScanAdapter
from transformation.adapters.lesion_segmentation_adapter import LesionSegmentationAdapter

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