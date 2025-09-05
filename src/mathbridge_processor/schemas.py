from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum


class SREDomain(str, Enum):
    MATHSPEAK = "mathspeak"
    CLEARSPEAK = "clearspeak"


class ProcessingConfig(BaseModel):
    sre_domain: SREDomain = SREDomain.CLEARSPEAK
    sre_locale: str = "en"
    batch_size: int = 1000
    max_records: Optional[int] = None
    resume_from: int = 0
    output_path: str = "mathbridge_processed"
    latex2sre_path: str = "./latex2sre"
    max_workers: Optional[int] = None  # None = auto-detect optimal workers


class ProcessingStats(BaseModel):
    total_processed: int = 0
    valid_latex: int = 0
    invalid_latex: int = 0
    speech_generated: int = 0
    speech_failed: int = 0


class ProcessingResult(BaseModel):
    config: ProcessingConfig
    stats: ProcessingStats
    output_files: List[str]
    errors: List[str] = []
    success: bool = True
    cache_stats: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
