from typing import Dict, List, Optional, Type, Tuple
from abc import ABC, abstractmethod
import math

RawCount = int
Weight = float
DocumentId = str

TermCountDict = Dict[str, RawCount]
TermWeightDict = Dict[str, Weight]

# TF Schemes
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
        return 0.5 + 0.5 * (count / max_count) if max_count else 0

# IDF Schemes
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
        return math.log(doc_count / term_count) if term_count > 0 else 0.0

class IdfSmoothScheme(IdfScheme):
    """Smooth inverse document frequency weighting scheme."""
    @staticmethod
    def calculate(doc_count: int, term_count: int, max_count: Optional[int] = None) -> float:
        return math.log((doc_count + 1) / (term_count + 1))

class IdfMaxScheme(IdfScheme):
    """Maximum inverse document frequency weighting scheme."""
    @staticmethod
    def calculate(doc_count: int, term_count: int, max_count: int) -> float:
        return math.log(max_count / (term_count + 1)) if term_count > 0 else 0.0

class IdfProbScheme(IdfScheme):
    """Probabilistic inverse document frequency weighting scheme."""
    @staticmethod
    def calculate(doc_count: int, term_count: int, max_count: Optional[int] = None) -> float:
        return math.log((doc_count - term_count) / term_count) if term_count > 0 else 0.0

# TF-IDF Schemes
class TfIdfScheme(ABC):
    """Abstract base class for combined TF-IDF weighting schemes."""
    @abstractmethod
    @staticmethod
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

class Document:
    def __init__(self, text: str, doc_id: DocumentId, default_tf_scheme: Type[TfScheme] = TfStandardScheme):
        self.text: str = text
        self.doc_id: DocumentId = doc_id
        self.raw_frequencies: TermCountDict = self._count_raw_frequencies()
        self.total_terms: int = sum(self.raw_frequencies.values())
        self.default_tf_scheme: Type[TfScheme] = default_tf_scheme
        #self.tf_cache: Dict[Type[TfScheme], TermWeightDict] = dict()

    def _count_raw_frequencies(self) -> TermCountDict:
        raw_frequencies: TermCountDict = dict()
        terms = self.text.split()

        for term in terms:
            raw_frequencies[term] = raw_frequencies.get(term, 0) + 1

        return raw_frequencies

    def get_term_frequency(self, tf_scheme: Optional[Type[TfScheme]] = None) -> TermWeightDict:
        """Calculate term frequency based on the specified weighting scheme."""
        if tf_scheme is None:
            tf_scheme = self.default_tf_scheme

        term_frequencies: TermWeightDict = dict()
        max_count = max(self.raw_frequencies.values(), default=1)

        for term, count in self.raw_frequencies.items():
            term_frequencies[term] = tf_scheme.calculate(count, self.total_terms, max_count)

        return term_frequencies

class Corpus:
    def __init__(self, documents: List[Document], default_idf_scheme: Type[IdfScheme] = IdfStandardScheme):
        self.documents: List[Document] = documents
        self.doc_count: int = len(documents)
        self.document_frequencies: TermCountDict = self._count_document_frequencies()
        self.default_idf_scheme: Type[IdfScheme] = default_idf_scheme

    def _count_document_frequencies(self) -> TermCountDict:
        """Count how many documents contain each term (document frequency)."""
        document_frequencies: TermCountDict = dict()

        for doc in self.documents:
            for term in doc.raw_frequencies:
                document_frequencies[term] = document_frequencies.get(term, 0) + 1

        return document_frequencies

    def get_inverse_document_frequency(self, idf_scheme: Optional[Type[IdfScheme]] = None) -> TermWeightDict:
        """Calculate inverse document frequency."""
        if idf_scheme is None:
            idf_scheme = self.default_idf_scheme

        inverse_document_frequencies: TermWeightDict = dict()

        max_count = max(self.document_frequencies.values(), default=1)

        for term, term_count in self.document_frequencies.items():
            inverse_document_frequencies[term] = idf_scheme.calculate(self.doc_count, term_count, max_count)

        return inverse_document_frequencies

class TfIdf:
    def __init__(self, corpus: Corpus):
        self.corpus: Corpus = corpus

    def calculate_scores(self, tfidf_scheme: Optional[Type[TfIdfScheme]] = None, tf_scheme: Optional[Type[TfScheme]] = None, idf_scheme: Optional[Type[IdfScheme]] = None) -> Dict[DocumentId, TermWeightDict]:
        tfidf_scores: Dict[DocumentId, TermWeightDict] = dict()

        if tfidf_scheme is not None:
            tf_scheme, idf_scheme = tfidf_scheme.get_schemes()

        idf = self.corpus.get_inverse_document_frequency(idf_scheme)
        
        for document in self.corpus.documents:
            tfidf_score: TermWeightDict = dict()
            for term, tf in document.get_term_frequency(tf_scheme).items():
                tfidf_score[term] = tf * idf[term]

        return tfidf_scores
