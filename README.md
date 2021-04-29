# EACL2021-DravidianTask-IITT
This repo contains implementation for the paper <b>"IIITT@DravidianLangTech-EACL2021: Transfer Learning for Offensive Language Detection in Dravidian Languages"</b> to be presented at EACL-2021 

In this paper, we propose to use several pretrained multilingual models (transformers) embeddings, then to be fed into a Bidirectional LSTM Layers.

To replicate the results, run the Colab files.
If you find this repo useful, please cite our paper :
```
@inproceedings{yasaswini-etal-2021-iiitt,
    title = "{IIITT}@{D}ravidian{L}ang{T}ech-{EACL}2021: Transfer Learning for Offensive Language Detection in {D}ravidian Languages",
    author = "Yasaswini, Konthala  and
      Puranik, Karthik  and
      Hande, Adeep  and
      Priyadharshini, Ruba  and
      Thavareesan, Sajeetha  and
      Chakravarthi, Bharathi Raja",
    booktitle = "Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages",
    month = apr,
    year = "2021",
    address = "Kyiv",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.dravidianlangtech-1.25",
    pages = "187--194",
    abstract = "This paper demonstrates our work for the shared task on Offensive Language Identification in Dravidian Languages-EACL 2021. Offensive language detection in the various social media platforms was identified previously. But with the increase in diversity of users, there is a need to identify the offensive language in multilingual posts that are largely code-mixed or written in a non-native script. We approach this challenge with various transfer learning-based models to classify a given post or comment in Dravidian languages (Malayalam, Tamil, and Kannada) into 6 categories. The source codes for our systems are published.",
}
```
