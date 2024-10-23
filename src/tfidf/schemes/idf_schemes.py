from typing import Optional
from abc import ABC, abstractmethod
import math

class IdfScheme(ABC):
    """Abstract base class for IDF weighting schemes."""
    @staticmethod
    @abstractmethod
    def calculate(doc_count: int, term_count: int, max_count: int) -> float:
        pass

class IdfUnaryScheme(IdfScheme):
    """Unary weighting scheme."""
    @staticmethod
    def calculate(doc_count: Optional[int] = None, term_count: Optional[int] = None, max_count: Optional[int] = None) -> float:
        return 1.0

class IdfStandardScheme(IdfScheme):
    """Standard inverse document frequency weighting scheme."""
    @staticmethod
    def calculate(doc_count: int, term_count: int, max_count: Optional[int] = None) -> float:
        return math.log(float(doc_count) / term_count) if term_count > 0 else 0.0

class IdfSmoothScheme(IdfScheme):
    """Smooth inverse document frequency weighting scheme."""
    @staticmethod
    def calculate(doc_count: int, term_count: int, max_count: Optional[int] = None) -> float:
        return math.log(float(doc_count + 1) / (term_count + 1))

class IdfMaxScheme(IdfScheme):
    """Maximum inverse document frequency weighting scheme."""
    @staticmethod
    def calculate(doc_count: int, term_count: int, max_count: int) -> float:
        return math.log(float(max_count) / (term_count + 1)) if term_count > 0 else 0.0

class IdfProbScheme(IdfScheme):
    """Probabilistic inverse document frequency weighting scheme."""
    @staticmethod
    def calculate(doc_count: int, term_count: int, max_count: Optional[int] = None) -> float:
        return math.log(float(doc_count - term_count) / term_count) if (term_count > 0 and doc_count - term_count > 0) else 0.0
