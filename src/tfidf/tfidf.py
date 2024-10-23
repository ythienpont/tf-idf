from typing import Dict, Optional, Type

from .schemes.tf_schemes import TfScheme
from .schemes.idf_schemes import IdfScheme
from .schemes.tfidf_schemes import TfIdfScheme
from .corpus import Corpus
from .types import DocumentId, TermWeightDict

class TfIdf:
    def __init__(self, corpus: Corpus):
        self.corpus: Corpus = corpus

    def calculate_scores(self, tfidf_scheme: Optional[Type[TfIdfScheme]] = None, tf_scheme: Optional[Type[TfScheme]] = None, idf_scheme: Optional[Type[IdfScheme]] = None) -> Dict[DocumentId, TermWeightDict]:
        scores: Dict[DocumentId, TermWeightDict] = dict()
        
        if tfidf_scheme is not None:
            tf_scheme, idf_scheme = tfidf_scheme.get_schemes()

        idf = self.corpus.get_inverse_document_frequency(idf_scheme)
        
        for doc in self.corpus.documents:
            tfidf_score: TermWeightDict = dict()
            for term, tf in doc.get_term_frequency(tf_scheme).items():
                tfidf_score[term] = tf * idf[term]
            
            scores[doc.doc_id] = tfidf_score

        return scores
