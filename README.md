# UNSCon: The UN Security Council Conflict Corpus

Dataset, annotation guidelines and code for classification experiments for "How Diplomats Dispute: The UN Security Council Conflict Corpus", by Karolina Zaczynska, Peter Bourgonje and Manfred Stede, to appear in the proceedings of LREC-COLING 2024.

#### For Lexicoder sentiment analysis: 
- download the the Lexicoder Sentiment Dictionary ("LSDaug2015") from the tool's website: https://www.snsoroka.com/data-lexicoder/
- save unzipped folder in /lexicoder_UNSC/data
- run make_lsd.py for preparing the lexicon for further processing (eventually update paths in config.py)
- run `LSD_sentiments_per_Sents.py` for sentiment classifications per sentence 

#### For bert-classifications:
-  run `bert-cl/sequence_labeling.py -c *path_to_csv*`. Select the relevant model on lines 23-25. The script wil create a folder with fine-tuned models and prints the evaluation scores.


The content in this repository can be used under CC BY-SA license, which allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.
