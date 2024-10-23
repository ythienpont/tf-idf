from abc import ABC, abstractmethod
from typing import Tuple, Type

from .tf_schemes import TfScheme, TfStandardScheme, TfRawScheme, TfDoubleNormScheme, TfLogNormScheme
from .idf_schemes import IdfScheme, IdfStandardScheme

class TfIdfScheme(ABC):
    """Abstract base class for combined TF-IDF weighting schemes."""
    @staticmethod
    @abstractmethod
    def get_schemes() -> Tuple[Type[TfScheme], Type[IdfScheme]]:
        pass

class TfIdfStandardScheme(TfIdfScheme):
    """Combined TF (Standard) and IDF (Standard) scheme."""
    @staticmethod
    def get_schemes() -> Tuple[Type[TfScheme], Type[IdfScheme]]:
        return TfStandardScheme, IdfStandardScheme

class TfIdfRawStandardScheme(TfIdfScheme):
    """Combined TF (Raw) and IDF (Standard) scheme."""
    @staticmethod
    def get_schemes() -> Tuple[Type[TfScheme], Type[IdfScheme]]:
        return TfRawScheme, IdfStandardScheme

class TfIdfDoubleNormScheme(TfIdfScheme):
    """Combined TF (Double Normalization) and IDF (Standard) scheme."""
    @staticmethod
    def get_schemes() -> Tuple[Type[TfScheme], Type[IdfScheme]]:
        return TfDoubleNormScheme, IdfStandardScheme

class TfIdfLogNormScheme(TfIdfScheme):
    """Combined TF (Log Normalization) and IDF (Standard) scheme."""
    @staticmethod
    def get_schemes() -> Tuple[Type[TfScheme], Type[IdfScheme]]:
        return TfLogNormScheme, IdfStandardScheme
