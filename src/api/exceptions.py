"""
Custom exceptions for the Automated RCA System.
"""


class RCAException(Exception):
    def __init__(self, message: str, code: str = "RCA_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class IncidentNotFoundError(RCAException):
    def __init__(self, incident_id: str):
        super().__init__(
            message=f"Incident '{incident_id}' not found",
            code="INCIDENT_NOT_FOUND",
        )
        self.incident_id = incident_id


class PipelineError(RCAException):
    def __init__(self, message: str, stage: str):
        super().__init__(
            message=f"Pipeline error in stage '{stage}': {message}",
            code="PIPELINE_ERROR",
        )
        self.stage = stage


class ModelLoadError(RCAException):
    def __init__(self, model_name: str):
        super().__init__(
            message=f"Failed to load model '{model_name}'",
            code="MODEL_LOAD_ERROR",
        )
        self.model_name = model_name
