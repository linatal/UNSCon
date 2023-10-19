import re
import os
import csv
import sys
from tqdm import tqdm
from optparse import OptionParser
from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import logging
import evaluate
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
os.environ["WANDB_DISABLED"] = "true"
logging.set_verbosity_info()

lid = 0
id2label = {}
label2id = {}

HF_MODEL = "distilbert-base-uncased"
#HF_MODEL = "chkla/roberta-argument"
#HF_MODEL = "siebert/sentiment-roberta-large-english"
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")


def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def get_seq_labels(csvf):
    edus = []
    a0_labels = []
    a2_labels = []
    a3_labels = []
    a4_labels = []
    b1_labels = []
    b2_labels = []
    b3_labels = []
    c2_labels = []
    c3_labels = []

    global lid
    rows = csv.reader(open(csvf), delimiter=',')
    headers = next(rows)
    for row in rows:
        edu = row[1]
        edus.append(edu)
        """
        a0 = row[4]
        a2 = row[5]
        a3 = row[6]
        a4 = row[7]
        b1 = row[8]
        b2 = row[9]
        b3 = row[10]
        """
        c2 = row[13]
        c3 = row[14]
        """
        if a0 not in label2id:
            id2label[lid] = a0
            label2id[a0] = lid
            lid += 1
        if a2 not in label2id:
            id2label[lid] = a2
            label2id[a2] = lid
            lid += 1
        if a3 not in label2id:
            id2label[lid] = a3
            label2id[a3] = lid
            lid += 1
        if a4 not in label2id:
            id2label[lid] = a4
            label2id[a4] = lid
            lid += 1
        if b1 not in label2id:
            id2label[lid] = b1
            label2id[b1] = lid
            lid += 1
        if b2 not in label2id:
            id2label[lid] = b2
            label2id[b2] = lid
            lid += 1
        if b3 not in label2id:
            id2label[lid] = b3
            label2id[b3] = lid
            lid += 1
        """
        if c2 not in label2id:
            id2label[lid] = c2
            label2id[c2] = lid
            lid += 1
        if c3 not in label2id:
            id2label[lid] = c3
            label2id[c3] = lid
            lid += 1

        """
        a0_labels.append(label2id[a0])
        a2_labels.append(label2id[a2])
        a3_labels.append(label2id[a3])
        a4_labels.append(label2id[a4])
        b1_labels.append(label2id[b1])
        b2_labels.append(label2id[b2])
        b3_labels.append(label2id[b3])
        """
        c2_labels.append(label2id[c2])
        c3_labels.append(label2id[c3])

    #labels = [a0_labels, a2_labels, a3_labels, a4_labels, b1_labels, b2_labels, b3_labels, c2_labels, c3_labels]
    labels = [c2_labels, c3_labels]

    return edus, labels


def get_splits(iterations, size):

    p = int(size / iterations)
    pl = [int(x) for x in range(0, size, p)]
    pl.append(int(size))
    return pl

def get_latest_checkpoint(cfolder):
    
    checkpoints = []
    for sf in os.listdir(cfolder):
        if sf.startswith('checkpoint-'):
            checkpoints.append(int(re.sub('checkpoint-', '', sf)))
    return max(checkpoints)

def cross_validate(edus, all_labels):
    n = 10
    splits = get_splits(n, len(edus))
    #label_types = ['a0', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3']
     
    label_types = ['binary_setup', '3class_setup']
    results = {k: {'trained_f1_micro': 0,
                   'trained_precision_micro': 0,
                   'trained_recall_micro': 0,
                   'majority_vote_f1_micro': 0,
                   'majority_vote_precision_micro': 0,
                   'majority_vote_recall_micro': 0,
                   'trained_f1_macro': 0,
                   'trained_precision_macro': 0,
                   'trained_recall_macro': 0,
                   'majority_vote_f1_macro': 0,
                   'majority_vote_precision_macro': 0,
                   'majority_vote_recall_macro': 0,
                   'trained_f1_weighted': 0,
                   'trained_precision_weighted': 0,
                   'trained_recall_weighted': 0,
                   'majority_vote_f1_weighted': 0,
                   'majority_vote_precision_weighted': 0,
                   'majority_vote_recall_weighted': 0
                   }
               for k in label_types}
    root_outdir = 'classifier_models_%s' % HF_MODEL
    if not os.path.exists(root_outdir):
        os.mkdir(root_outdir)

    for ji, label_type in enumerate(label_types):
        sys.stderr.write('INFO: Starting %i-fold cross validation for label %s.\n' % (n, label_type))
        labels = all_labels[ji]
        all_inputs = []
        all_preds = []
        all_golds = []

        for i in tqdm(range(n), desc='cross-validating...'):
            X_train = []
            y_train = []
            X_test = []
            y_test = []
            for index, tupl in enumerate(zip(edus, labels)):
                edu, label = tupl
                if splits[i] <= index <= splits[i + 1]:
                    X_test.append(edu)
                    y_test.append(label)
                else:
                    X_train.append(edu)
                    y_train.append(label)
            d = DatasetDict(
                {'train': Dataset.from_list([{'text': p[0], 'label': p[1]} for p in zip(X_train, y_train)]),
                 'test': Dataset.from_list([{'text': p[0], 'label': p[1]} for p in zip(X_test, y_test)])}
            )
            tokenized_d = d.map(preprocess_function, batched=True)
            #"""
            model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL, num_labels=len(label2id),
                                                                       id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
            training_args = TrainingArguments(
                output_dir=os.path.join(root_outdir, "%s_%i" % (label_type, i)),
                learning_rate=1e-5,#2e-5
                per_device_train_batch_size=32,#16
                per_device_eval_batch_size=32,#16
                num_train_epochs=2,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                push_to_hub=False,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_d["train"],
                eval_dataset=tokenized_d["test"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )

            trainer.train()
            #"""

            latest_checkpoint = get_latest_checkpoint(os.path.join(root_outdir, '%s_%i' % (label_type, i)))
            pred_tokenizer = AutoTokenizer.from_pretrained(os.path.join(root_outdir, "%s_%i" % (label_type, i), "checkpoint-%i" % latest_checkpoint))
            pred_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(root_outdir, "%s_%i" % (label_type, i), "checkpoint-%i" % latest_checkpoint), ignore_mismatched_sizes=True)

            for item in zip(X_test, y_test):
                edu, gold = item
                all_inputs.append(edu)
                inputs = pred_tokenizer(edu, return_tensors="pt")
                with torch.no_grad():
                    logits = pred_model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                pred = pred_model.config.id2label[predicted_class_id]
                all_preds.append(pred)
                all_golds.append(gold)
        assert len(all_inputs) == len(all_golds)
        label_debug_out = csv.writer(open('%s_outputs.txt' % label_type, 'w', newline='', encoding='utf8'), delimiter='\t')
        all_preds = [label2id[x] for x in all_preds]
        label_debug_out.writerow(['input', 'gold_label', 'pred_label'])
        for ii in range(len(all_inputs)):
            inp = all_inputs[ii]
            gold = all_golds[ii]
            pred = all_preds[ii]
            label_debug_out.writerow([inp, id2label[gold], id2label[pred]])
        trained_f1_micro = f1_score(all_golds, all_preds, average='micro')
        trained_precision_micro = precision_score(all_golds, all_preds, average='micro')
        trained_recall_micro = recall_score(all_golds, all_preds, average='micro')
        trained_f1_macro = f1_score(all_golds, all_preds, average='macro')
        trained_precision_macro = precision_score(all_golds, all_preds, average='macro')
        trained_recall_macro = recall_score(all_golds, all_preds, average='macro')
        trained_f1_weighted = f1_score(all_golds, all_preds, average='weighted')
        trained_precision_weighted = precision_score(all_golds, all_preds, average='weighted')
        trained_recall_weighted = recall_score(all_golds, all_preds, average='weighted')
        #print('f1-score for %i-fold cross-validation for %s: %f' % (n, label_type, trained_f1_micro))
        mv = max(set(all_golds), key=all_golds.count)
        pred_mv = [mv] * len(all_golds)
        mv_f1_micro = f1_score(all_golds, pred_mv, average='micro')
        mv_precision_micro = precision_score(all_golds, pred_mv, average='micro')
        mv_recall_micro = recall_score(all_golds, pred_mv, average='micro')
        mv_f1_macro = f1_score(all_golds, pred_mv, average='macro')
        mv_precision_macro = precision_score(all_golds, pred_mv, average='macro')
        mv_recall_macro = recall_score(all_golds, pred_mv, average='macro')
        mv_f1_weighted = f1_score(all_golds, pred_mv, average='weighted')
        mv_precision_weighted = precision_score(all_golds, pred_mv, average='weighted')
        mv_recall_weighted = recall_score(all_golds, pred_mv, average='weighted')
        #print('majority vote classifier:', mv_f1)
        results[label_type]['trained_f1_micro'] = trained_f1_micro
        results[label_type]['trained_precision_micro'] = trained_precision_micro
        results[label_type]['trained_recall_micro'] = trained_recall_micro
        results[label_type]['trained_f1_macro'] = trained_f1_macro
        results[label_type]['trained_precision_macro'] = trained_precision_macro
        results[label_type]['trained_recall_macro'] = trained_recall_macro
        results[label_type]['trained_f1_weighted'] = trained_f1_weighted
        results[label_type]['trained_precision_weighted'] = trained_precision_weighted
        results[label_type]['trained_recall_weighted'] = trained_recall_weighted
        results[label_type]['mv_f1_micro'] = mv_f1_micro
        results[label_type]['mv_precision_micro'] = mv_precision_micro
        results[label_type]['mv_recall_micro'] = mv_recall_micro
        results[label_type]['mv_f1_macro'] = mv_f1_macro
        results[label_type]['mv_precision_macro'] = mv_precision_macro
        results[label_type]['mv_recall_macro'] = mv_recall_macro
        results[label_type]['mv_f1_weighted'] = mv_f1_weighted
        results[label_type]['mv_precision_weighted'] = mv_precision_weighted
        results[label_type]['mv_recall_weighted'] = mv_recall_weighted

        print('+++++++++++++++++ classification report for %s ++++++++++++++++++++++++++++' % label_type)
        print('id2label:',pred_model.config.id2label)
        print(classification_report(all_golds, all_preds))

    print('\n\n################## FINAL RESULTS ##################\n\n')
    for label_type in results:
        print('Label type:', label_type)
        print('\t classifier f1 micro:', results[label_type]['trained_f1_micro'])
        print('\t classifier precision micro:', results[label_type]['trained_precision_micro'])
        print('\t classifier recall micro:', results[label_type]['trained_recall_micro'])
        print('\t majority vote f1 micro:', results[label_type]['mv_f1_micro'])
        print('\t majority vote precision micro:', results[label_type]['mv_precision_micro'])
        print('\t majority vote recall micro:', results[label_type]['mv_recall_micro'])
        print('\t classifier f1 macro:', results[label_type]['trained_f1_macro'])
        print('\t classifier precision macro:', results[label_type]['trained_precision_macro'])
        print('\t classifier recall macro:', results[label_type]['trained_recall_macro'])
        print('\t majority vote f1 macro:', results[label_type]['mv_f1_macro'])
        print('\t majority vote precision macro:', results[label_type]['mv_precision_macro'])
        print('\t majority vote recall macro:', results[label_type]['mv_recall_macro'])
        print('\t classifier f1 weighted:', results[label_type]['trained_f1_weighted'])
        print('\t classifier precision weighted:', results[label_type]['trained_precision_weighted'])
        print('\t classifier recall weighted:', results[label_type]['trained_recall_weighted'])
        print('\t majority vote f1 weighted:', results[label_type]['mv_f1_weighted'])
        print('\t majority vote precision weighted:', results[label_type]['mv_precision_weighted'])
        print('\t majority vote recall weighted:', results[label_type]['mv_recall_weighted'])
        print()

    nr_edus = len(edus)
    nr_tokens = sum([len(t.split()) for t in edus])
    print('Results of %i-fold cross-validation over %i edus, containing %i tokens' % (n, nr_edus, nr_tokens))


def main():
    parser = OptionParser()
    parser.add_option("-c", "--csv", dest="csvfile", help="csv file with edus and labels")

    (options, args) = parser.parse_args()
    if not options.csvfile:
        parser.print_help()
        sys.exit()

    edus, all_labels = get_seq_labels(options.csvfile)
    cross_validate(edus, all_labels)


if __name__ == '__main__':
    main()
