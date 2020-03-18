import argparse
import json
from typing import List

from sklearn.metrics import f1_score

LANGUAGES = [
    "de",
    "fr",
    "it",
]

TEST_SETS = [
    "new_comments_defr",
    "new_questions_defr",
    "new_topics_defr",
    "new_comments_it",
    # "new_questions_it",
    # "new_topics_it",
]


def evaluate_file(gold_file, pred_file):
    gold_list = [json.loads(line) for line in gold_file]
    pred_list = [json.loads(line) for line in pred_file]
    args.gold.close()
    args.pred.close()
    evaluate_json(gold_list, pred_list)


def evaluate_json(gold: List, pred: List):
    for test_set in TEST_SETS:
        print(test_set)
        for language in LANGUAGES:
            instance_indices = [i for i, instance in enumerate(gold) if
                                instance["test_set"] == test_set and instance["language"] == language]
            gold_labels = [gold[i]["label"] for i in instance_indices]
            pred_labels = [pred[i]["label"] for i in instance_indices]
            if not len(gold_labels):
                continue
            score = f1_score(gold_labels, pred_labels, average="macro")
            print(language.upper(), 100 * score)
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate predictions on the x-stance test sets")
    parser.add_argument('--gold', type=argparse.FileType('r', encoding='UTF-8'), required=True)
    parser.add_argument('--pred', type=argparse.FileType('r', encoding='UTF-8'), required=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    evaluate_file(args.gold, args.pred)
