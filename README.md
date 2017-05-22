# startup-generator
Using Markov chains to generate fake startup ideas: the monkeys on typewriters approach to founding a company.

##### Some of our favorite examples:

* Anaphore develops protein therapeutics to treat diseases affecting dogs and breeding.
* Focus Photography Inc specializes in state-of-the-art functional brain imaging that utilizes AJAX technologies in the development of Android apps.
* The Australian soft toy for children and their patients.
* Firalis is a fashion-based crowdfunding platform for gamers.
* A123 Systems is an open community of genealogists collaborating to help users manage their bank accounts from one Web-based inbox.
* Jewelry for the iPhone.
* Good Health Media is an online platform that offers sports fans the opportunity to go on a fascinating wine adventure.

----
#### Quick start
```
pip install unidecode
python main.py
```

----
#### Setup
```
pip install nltk
pip install unidecode
pip install pattern
```

NB: if you want to do part-of-speech tagging yourself, you must have installed nltk's pos data:
```
python -m nltk.downloader maxent_treebank_pos_tagger
python -m nltk.downloader averaged_perceptron_tagger
```

----
#### Run
```
python main.py
```

----
#### Contributors
[Colton Gyulay](https://github.com/cgyulay)

[Antuan Tran](https://github.com/tuantrain)

[Evan Gastman](https://github.com/evangastman)
