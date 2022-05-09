# LDSI project
## WS 2021 - 2022 at TUM
### Hypothetical Scenario:
You are a data scientist on a consulting team hired by the US Board of Veteran's Appeals.
Your project is to develop a new search engine for BVA decisions. The goal is to produce a
system capable of processing decisions and extracting the reasons for why cases where
granted, denied, or remanded. The institution has asked you to survey relevant literature and
conduct a pilot experiment so that an informed decision can be made about how to invest
development resources. The analytical objective of the project is to test whether different
kinds of sentence annotations/markup for BVA decisions can be automated, what lessons can
be drawn regarding the difficulty of classifying certain types, and what challenges are there for
generally handling BVA decision texts in NLP pipelines.

### How to test the model?
You can test the model with a txt file. To do so you must write in the command line:
```
python analyze.py <path/file.txt>
```
There are also 2 optional arguments:
- "write_to_csv" (-csv): boolean: when False the output is printed in the terminal, 
when True the output is written in a csv file ('output_predicited_type.csv' in folder 'outputs') but no more in the terminal. Its default value is False.
- "model" (-m): string: The model you want to use for prediction. Either 'LR&TFIDF' for Logistic Regression with TF-IDF embeddings (default) or or 'LR' for Logistic Regression with Fasttext embeddings or 'RF' for Random Forest with Fasttext embeddings.
To use these optional arguments, you must write in the command line, for example:
```
python analyze.py <path/file.txt> -csv True -m 'LR'
```
The output has on each line a sentence and its predicted type.

### References
- [1](https://github.com/jsavelka/luima_sbd) (Savelka, Jaromir, Vern R. Walker, Matthias Grabmair and Kevin D. Ashley. "Sentence Boundary Detection in Adjudicatory Decisions in the United States." TAL 58.2 (2017))
- [2](https://github.com/facebookresearch/fastText) (P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov. "Enriching Word Vectors with Subword Information" TACL 5 (2017))
- Thanks to the [TUM Legal Tech Group](https://www.in.tum.de/legaltech/home) for the workshop