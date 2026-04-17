from pneumonia_segmentation.adapters.nifti_adapter import NiftiAdapter

class TransformationAdapterFactory:
    @staticmethod
    def get_adapter(source_type: str, path: str):
        source_type = source_type.upper()
        
        if source_type == "NIFTI":
            return NiftiAdapter(path)
        else:
            raise ValueError(f"Source type '{source_type}' is not supported by AdapterFactory.")