import pandas as pd
pd.options.mode.chained_assignment = None
import os
import re

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def preprocess(text: str):
    """Remove punctuation, transform to lower case."""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

def find_sentiment_entries(string, df_lsd):
    """Get the sentiment words in a input string.
    Check the words in the string that are contained in the sentiment dictionary.
    Return the string and the words and their polarity.
    The polarities are negative|positive|neg_negative|neg_positive.

    Args
    ----
    edu (str): input string
    entry (list): list with LSD lexEntry, nrOfTokens, isPrefix, polarity

    Return
    ------
    str, list(tuple(str, str)): the preproc sentence, the sentiment words and their polarities.
    """
    out_entries = []
    edu_pp = preprocess(string)
    edu_pp_separated = edu_pp.split()

    for row in df_lsd.values:
        entry = row.tolist()
        # lex entry is single token
        if entry[1] == 1:
            # lex entry is not a prefix
            if entry[2] == 0:
                out_entries.extend([(entry[0], entry[3]) for t in edu_pp_separated if entry[0] == t])
            else:
            # if lex entry is a prefix
                out_entries.extend([(entry[0], entry[3]) for t in edu_pp_separated if t.startswith(entry[0])])
        else:
            # lex entry is not a prefix
            if entry[2] == 0:
                pattern_in_sent = entry[0] + " "
                out_entries.extend([(entry[0], entry[3]) for i in range((len(edu_pp.split(pattern_in_sent)) - 1))])
                if edu_pp.endswith(entry[0]):
                    out_entries.append((entry[0], entry[3]))
                else:
                    out_entries.extend([(entry[0], entry[3]) for i in range((len(edu_pp.split(entry[0])) - 1))])

    return edu_pp, out_entries

def calc_sentiment_score(sentence: str, pol_words):
    """Calculate the sentiment score of a sentence.
    The score is in [-1,1] and score = (n_positive_words - n_negative_words) / n_words.

    Args
    ----
    sentence (str): The sentence.
    pol_words (list(tuple(str, str)) or list(str)): The words and their polarities or just polarites.

    Return
    ------
    float: The sentiment score.
    """
    neg = sum([1 for t in pol_words if "negative" in t])
    neg_neg = sum([1 for t in pol_words if "neg_negative" in t])
    pos = sum([1 for t in pol_words if "positive" in t])
    neg_pos = sum([1 for t in pol_words if "neg_positive" in t])
    pos = pos - neg_pos + neg_neg
    neg = neg - neg_neg + neg_pos
    score = pos - neg
    if score != 0:
        score = score / len(sentence.split())
    return pol_words, score


def categorigal_df(score_df, polarity_words, scores):
    # to categorical binarized data, 0 = NEG, 1 = Neutral or POS
    score_df['NE_score_binary'] = (score_df['A0'] != "_").astype(int).tolist()
    score_df['C_score_binary'] = (score_df['B1'] != "_").astype(int)
    all_score = score_df['NE_score_binary'] + score_df['C_score_binary']

    score_df['ALL_score_binary'] = score_df['NE_score_binary'] + score_df['C_score_binary']
    # if Correction and NE
    score_df['ALL_score_binary'][score_df['ALL_score_binary'] > 1] = 1

    score_df['lexicoder_words_polarity'] = polarity_words
    score_df['lexicoder_score'] = scores
    score_df['lexicoder_score_binary'] = (score_df['lexicoder_score'] < 0).astype(int)
    return score_df


def validation(cat_df, y_true, y_pred):


    print('Precision: %.3f' % precision_score(y_true, y_pred, average='macro'))
    print('Recall: %.3f' % recall_score(y_true, y_pred, average='macro'))
    print('F1 Score: %.3f' % f1_score(y_true, y_pred, average='macro'))
    print('Accuracy: %.3f' % accuracy_score(y_true, y_pred))

    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    #plt.show()
    return True


def main():
    UNSC_data = "../../data/conflict_annotations_sentences.csv"
    df = pd.read_csv(UNSC_data)
    LEXICODER_data = "./data/lsd.tsv"
    df_lsd = pd.read_csv(LEXICODER_data, sep="\t")

    pol_word_list = []
    score_list = []
    for line_text in df["sentence_text"]:
        line_preprocessed, out_entries = find_sentiment_entries(line_text, df_lsd)
        pol_words, score = calc_sentiment_score(line_preprocessed, out_entries)
        pol_word_list.append(pol_words)
        score_list.append(score)
    new_dataframe=df[["file_id", "sentence_text", "A0", "B1"]]

    categorical_dataframe = categorigal_df(new_dataframe, pol_word_list, score_list)
    output = "./data/lexicoder_output_sents.csv"
    #categorical_dataframe.to_csv(output, index=False)

    y_true = categorical_dataframe['ALL_score_binary'].tolist() # nur das hier ändern
    y_true_ne = categorical_dataframe['NE_score_binary'].tolist()
    y_pred = categorical_dataframe['lexicoder_score_binary'].tolist()

    print("Scores for NE + CC:\n")
    validation(categorical_dataframe, y_true, y_pred)
    print("\nScores for NE only:\n")
    validation(categorical_dataframe, y_true_ne, y_pred)


if __name__ == "__main__":
    main()