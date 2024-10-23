from typing import Optional
from abc import ABC, abstractmethod
import math

class TfScheme(ABC):
    """Abstract base class for TF weighting schemes."""
    @staticmethod
    @abstractmethod
    def calculate(count: int, total: int, max_count: int) -> float:
        pass

class TfBinaryScheme(TfScheme):
    """Binary weighting scheme."""
    @staticmethod
    def calculate(count: int, total: Optional[int] = None, max_count: Optional[int] = None) -> float:
        return float(1 if count > 0 else 0)

class TfRawScheme(TfScheme):
    """Raw count weighting scheme."""
    @staticmethod
    def calculate(count: int, total: Optional[int] = None, max_count: Optional[int] = None) -> float:
        return float(count)

class TfStandardScheme(TfScheme):
    """Term frequency weighting scheme."""
    @staticmethod
    def calculate(count: int, total: int, max_count: Optional[int] = None) -> float:
        return float(count) / total

class TfLogNormScheme(TfScheme):
    """Log normalization weighting scheme."""
    @staticmethod
    def calculate(count: int, total: int = 0, max_count: Optional[int] = None) -> float:
        return math.log(1 + count)

class TfDoubleNormScheme(TfScheme):
    """Double normalization weighting scheme with constant 0.5."""
    @staticmethod
    def calculate(count: int, total: Optional[int] = None, max_count: int = 0) -> float:
        return 0.5 + 0.5 * (count / max_count) if max_count else 0.0
