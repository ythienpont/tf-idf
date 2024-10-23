import pytest

from tfidf.corpus import Corpus
from tfidf.document import Document
from tfidf.schemes.idf_schemes import IdfStandardScheme

def test_corpus_idf():
    doc1 = Document(["hello", "world"], "doc1")
    doc2 = Document(["hello", "everyone"], "doc2")
    corpus = Corpus([doc1, doc2])
    
    idf = corpus.get_inverse_document_frequency(IdfStandardScheme)

    assert idf['hello'] == 0.0
    assert idf['world'] > 0.0
