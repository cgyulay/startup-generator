# startup-generator
Using Markov chains to generate fake startup ideas: the monkeys on typewriters approach to founding a company.

----
####Setup
```
pip install nltk
pip install unidecode
pip install pattern
python main.py
```

NB: if you want to do part-of-speech tagging yourself, you must have installed nltk's pos data:
```
python -m nltk.downloader maxent_treebank_pos_tagger
python -m nltk.downloader averaged_perceptron_tagger
```
