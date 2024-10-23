from typing import Optional, Type, List
from collections import Counter

from .schemes.tf_schemes import TfScheme, TfStandardScheme
from .types import DocumentId, TermCounter, TermWeightDict

class Document:
    def __init__(self, terms: List[str], doc_id: DocumentId, default_tf_scheme: Type[TfScheme] = TfStandardScheme):
        self.terms: List[str] = terms
        self.doc_id: DocumentId = doc_id
        self.raw_frequencies: TermCounter = Counter(terms)
        self.total_terms: int = len(terms)
        self.default_tf_scheme: Type[TfScheme] = default_tf_scheme

    def get_term_frequency(self, tf_scheme: Optional[Type[TfScheme]] = None) -> TermWeightDict:
        """Calculate term frequency based on the specified weighting scheme."""
        if tf_scheme is None:
            tf_scheme = self.default_tf_scheme

        term_frequencies: TermWeightDict = dict()
        max_count = max(self.raw_frequencies.values(), default=1)

        for term, count in self.raw_frequencies.items():
            term_frequencies[term] = tf_scheme.calculate(count, self.total_terms, max_count)

        return term_frequencies
