# 🌟 Simple TF-IDF Library

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-success)

## 📖 Overview

TF-IDF (Term Frequency-Inverse Document Frequency) is a numerical statistic used to reflect the importance of a term in a document relative to a collection (or corpus) of documents. It is widely used in information retrieval and text mining. 

This library is a hobby project aimed at understanding the basics of Natural Language Processing (NLP). It's designed to only provide basic functionality, so optimizations are for the user to implement.

## ⚙️ Features

- **TF-IDF Calculation**: Easily compute TF-IDF scores for documents.
- **Modular Design**: Easily integrate with text processing libraries.
- **Customization**: Supports various weighting schemes for TF and IDF.
  
## 🛠️ Usage
```python
    doc1 = Document(["hello", "world", "hello"], "doc1")
    doc2 = Document(["hello", "everyone"], "doc2")

    corpus = Corpus([doc1, doc2])

    tfidf = TfIdf(corpus)
    scores = tfidf.calculate_scores()

    print(scores["doc1"]["hello"])

    # It's also possible to use different weighting schemes
    idf_smooth = corpus.get_inverse_document_frequency(IdfSmoothScheme)
    tf = doc1.get_term_frequency(TfLogNormScheme)
```

## 🔮 Future Plans

If I feel like working on it some more, it will probably be on:

- **Caching**: Implement caching to avoid recalculating scores.
- **Incremental Updates**: Allow users to update the corpus without recalculating everything.
- **Data Export**: Add functionality to export computed scores for further analysis.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.”

## 🤝 Contributing 
Contributions are welcome! Feel free to open issues or submit pull requests.
