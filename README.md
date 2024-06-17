# UNSCon: The UN Security Council Conflict Corpus

Dataset, annotation guidelines and code for classification experiments for the paper "How Diplomats Dispute: The UN Security Council Conflict Corpus" by Karolina Zaczynska, Peter Bourgonje and Manfred Stede. Appeared in Proceedings of the Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024). Turin, Italy, 2024. URL: https://aclanthology.org/2024.lrec-main.716.pdf


### Dataset
Annotated UNSCon corpus as csv-file with EDU-split speeches and conflict-labels: `conflict_annotations.csv`

### Classification Experiments
#### For Lexicoder sentiment analysis: 
- Download the the Lexicoder Sentiment Dictionary ("LSDaug2015") from the tool's website: https://www.snsoroka.com/data-lexicoder/
- Unzip folder to /lexicoder_UNSC/data/ (`unzip $PATH/LDSDaug2015.zip -d ./lexicoder_UNSC/data`)
- Run `make_lsd.py` to prepare the lexicon for further processing (eventually update paths in config.ini)
- Run `LSD_sentiments_per_Sents.py` for sentiment classifications per sentence 

#### For bert-classifications:
-  Run classification experiments with `bert-cl/sequence_labeling.py -c ./data/conflict_annotations.csv`.  
Select the relevant model on lines 23-25. The script wil create a folder with fine-tuned models and prints the evaluation scores.



The content in this repository can be used under CC BY-SA license, which allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.
