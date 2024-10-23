from tfidf.corpus import Corpus
from tfidf.document import Document
from tfidf.schemes.idf_schemes import IdfStandardScheme
from tfidf.schemes.tf_schemes import TfRawScheme
from tfidf.tfidf import TfIdf

def test_tfidf_calculation():
    # Create sample documents
    doc1 = Document(["hello", "world", "hello"], "doc1")
    doc2 = Document(["hello", "everyone"], "doc2")
    corpus = Corpus([doc1, doc2])
    
    # Create TfIdf instance
    tfidf = TfIdf(corpus)

    # Calculate TF-IDF scores
    scores = tfidf.calculate_scores()

    # Get expected IDF values
    idf = corpus.get_inverse_document_frequency(IdfStandardScheme)
    
    assert scores["doc1"]["hello"] == ((2.0/3.0) * idf["hello"])
    assert scores["doc1"]["world"] == ((1.0/3.0) * idf["world"])
    assert scores["doc2"]["hello"] == (0.5 * idf["hello"])
    assert scores["doc2"]["everyone"] == (0.5 * idf["everyone"])

    assert scores["doc1"]["hello"] < scores["doc1"]["world"]
    assert scores["doc1"]["world"] < scores["doc2"]["everyone"]

def test_tfidf_with_custom_schemes():
    # Create sample documents
    doc1 = Document(["test", "example"], "doc1")
    doc2 = Document(["test", "sample"], "doc2")
    corpus = Corpus([doc1, doc2])
    
    # Create TfIdf instance
    tfidf = TfIdf(corpus)

    scores = tfidf.calculate_scores(tf_scheme=TfRawScheme, idf_scheme=IdfStandardScheme)

    idf = corpus.get_inverse_document_frequency(IdfStandardScheme)

    assert scores["doc1"]["test"] == (1.0 * idf["test"])
    assert scores["doc1"]["example"] == (1.0 * idf["example"])
    assert scores["doc2"]["test"] == (1.0 * idf["test"])
    assert scores["doc2"]["sample"] == (1.0 * idf["sample"])
