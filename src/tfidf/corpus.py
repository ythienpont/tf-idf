from typing import List, Type, Optional
from collections import Counter

from .document import Document
from .schemes.idf_schemes import IdfScheme, IdfStandardScheme
from .types import TermCounter, TermWeightDict

class Corpus:
    def __init__(self, documents: List[Document], default_idf_scheme: Type[IdfScheme] = IdfStandardScheme):
        self.documents: List[Document] = documents
        self.doc_count: int = len(documents)
        self.document_frequencies: TermCounter = self._count_document_frequencies()
        self.default_idf_scheme: Type[IdfScheme] = default_idf_scheme

    def _count_document_frequencies(self) -> TermCounter:
        """Count how many documents contain each term (document frequency)."""
        document_frequencies: TermCounter = Counter()

        for doc in self.documents:
            document_frequencies.update(set(doc.raw_frequencies.keys()))

        return document_frequencies

    def get_inverse_document_frequency(self, idf_scheme: Optional[Type[IdfScheme]] = None) -> TermWeightDict:
        """Calculate inverse document frequency."""
        if idf_scheme is None:
            idf_scheme = self.default_idf_scheme

        inverse_document_frequencies: TermWeightDict = dict()

        most_common_element = self.document_frequencies.most_common(1)
        _, max_count = most_common_element[0] if most_common_element else (None, 0)

        for term, term_count in self.document_frequencies.items():
            inverse_document_frequencies[term] = idf_scheme.calculate(self.doc_count, term_count, max_count)

        return inverse_document_frequencies
