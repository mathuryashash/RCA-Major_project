"""
Error response models for the Automated RCA System.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    code: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str
