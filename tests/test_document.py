from tfidf.document import Document
from tfidf.schemes.tf_schemes import TfRawScheme

def test_document_term_frequency():
    terms = ["hello", "world", "hello"]
    doc_id = "doc1"
    document = Document(terms,doc_id)

    tf = document.get_term_frequency(TfRawScheme)

    assert tf["hello"] == 2.0
    assert tf["world"] == 1.0
