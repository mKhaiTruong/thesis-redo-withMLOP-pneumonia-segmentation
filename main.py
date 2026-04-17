import os, sys

from pneumonia_segmentation import logging
from pneumonia_segmentation.exception import CustomException

from pneumonia_segmentation.pipeline.stage1_data_ingestion import DataIngestionPipeline
# from pneumonia_segmentation.pipeline.stage2_data_transformation import DataTransformationPipeline
# from pneumonia_segmentation.pipeline.stage3_data_drift import DataDriftPipeline
# from pneumonia_segmentation.pipeline.stage4_prepare_base_model import PrepareBaseModelPipeline
# from pneumonia_segmentation.pipeline.stage5_training import TrainingPipeline
# from pneumonia_segmentation.pipeline.stage6_onnx import OnnxPipeline
# from pneumonia_segmentation.pipeline.stage7_tensorRT import TensorRTPipeline
# from pneumonia_segmentation.pipeline.stage8_evaluation import EvaluationPipeline

if __name__ == "__main__":
        STAGE_NAME = "Data Ingestion Stage"
        try:
                logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
                data_ingestion = DataIngestionPipeline()
                data_ingestion.main()
                logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        except Exception as e:
                raise CustomException(e, sys)

        # STAGE_NAME = "Data Transformation Stage"
        # try:
        #         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #         data_transformation = DataTransformationPipeline()
        #         data_transformation.main()
        #         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        # except Exception as e:
        #         raise CustomException(e, sys)
        
        # STAGE_NAME = "Data drift Stage"
        # try:
        #         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #         data_drift = DataDriftPipeline()
        #         data_drift.main()
        #         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        # except Exception as e:
        #         raise CustomException(e, sys)

        # STAGE_NAME = "Prepare Base Model Stage"
        # try:
        #         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #         prepare_base_model = PrepareBaseModelPipeline()
        #         prepare_base_model.main()
        #         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        # except Exception as e:
        #         raise CustomException(e, sys)
        
        # STAGE_NAME = "Training Stage"
        # try:
        #         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #         training = TrainingPipeline()
        #         training.main()
        #         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        # except Exception as e:
        #         raise CustomException(e, sys)
        
        # STAGE_NAME = "ONNX Stage"
        # try:
        #         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #         onnx = OnnxPipeline()
        #         onnx.main()
        #         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        # except Exception as e:
        #         raise CustomException(e, sys)
        
        # STAGE_NAME = "TensorRT Stage"
        # try:
        #         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #         tensorRT = TensorRTPipeline()
        #         tensorRT.main()
        #         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=========x")
        # except Exception as e:
        #         raise CustomException(e, sys) 
        
        # STAGE_NAME = "Evaluation Stage"
        # try:
        #         logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        #         evaluation = EvaluationPipeline()
        #         evaluation.main()
        #         logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        # except Exception as e:
        #         raise CustomException(e, sys)
        