import argparse
import os
import subprocess
from typing import Tuple

import jsonlines
import fasttext
from numpy.core.multiarray import ndarray


def _jsonl_to_fasttext_format(input_path: str, output_path: str) -> None:
    # Basic preprocessing and split into language files
    language_order = []
    with jsonlines.open(input_path) as f_in, \
            open(output_path + ".de", "w") as f_out_de, \
            open(output_path + ".fr", "w") as f_out_fr, \
            open(output_path + ".it", "w") as f_out_it:
        for line in f_in:
            # Concatenate question and comment
            text = " ".join([line["question"], line["comment"]])
            text = text.replace("\n", " ")
            language = line["language"]
            language_order.append(language)
            if language == "de":
                output_file = f_out_de
            elif language == "fr":
                output_file = f_out_fr
            elif language == "it":
                output_file = f_out_it
            else:
                raise NotImplementedError()
            output_file.write("__label__{} {}\n".format(line["label"], text))
    # Language-specific tokenization
    for language in ["de", "fr", "it"]:
        with open(output_path + ".{}".format(language)) as f_in, \
                open(output_path + ".tokenized.{}".format(language), "w") as f_out:
            subprocess.call([
                    "./tools/tokenizer.perl",
                    "-l", language,
                    "-q",
                ], stdin=f_in, stdout=f_out,
            )
    # Merge language files
    with open(output_path, "w") as f_out, \
            open(output_path + ".tokenized.de") as f_in_de, \
            open(output_path + ".tokenized.fr") as f_in_fr, \
            open(output_path + ".tokenized.it") as f_in_it:
        for language in language_order:
            if language == "de":
                output_file = f_in_de
            elif language == "fr":
                output_file = f_in_fr
            else:
                output_file = f_in_it
            line = next(output_file)
            line = line.replace("_ _ label _ _ ", "__label__")
            f_out.write(line)


def _predictions_to_jsonl(predictions: Tuple[Tuple[Tuple[str], ndarray]], output_path: str) -> None:
    with jsonlines.open(output_path, "w") as f:
        for labels, _ in predictions:
            label = labels[0].replace("__label__", "")
            f.write({"label": label})


def train(model_path: str, train_dataset_path: str, pretrained_vectors: str = "", lr: float = 0.1,
          epochs: int = 5) -> fasttext.FastText:
    model = fasttext.train_supervised(
        input=train_dataset_path,
        pretrained_vectors=pretrained_vectors,
        dim=300,
        lr=lr,
        epoch=epochs,
        wordNgrams=3,
    )
    model.save_model(model_path)
    return model


def predict(model_path: str, dataset_path) -> Tuple[Tuple[Tuple[str], ndarray]]:
    model = fasttext.load_model(model_path)
    predictions = []
    with open(dataset_path) as f:
        for line in f:
            if not line.strip():
                continue
            _, *tokens = line.split()
            text = " ".join(tokens)
            prediction: Tuple[Tuple[str], ndarray] = model.predict(text)
            predictions.append(prediction)
    return tuple(predictions)


def main():
    parser = argparse.ArgumentParser(description="Train a fastText baseline for X-Stance")
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--pred', type=str, required=True)
    parser.add_argument('--pretrained-vectors', type=str, default="")
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    FASTTEXT_DATA_DIR = "processed_data"
    for dataset_path in [
        "train.jsonl",
        "valid.jsonl",
        "test.jsonl",
    ]:
        _jsonl_to_fasttext_format(
            input_path=os.path.join(args.data_dir, dataset_path),
            output_path=os.path.join(FASTTEXT_DATA_DIR, dataset_path.replace(".jsonl", ".txt"))
        )

    model_path = "model.bin"
    model = train(
        model_path,
        train_dataset_path=os.path.join(FASTTEXT_DATA_DIR, "train.txt"),
        pretrained_vectors=args.pretrained_vectors,
        lr=args.lr,
        epochs=args.epochs,
    )
    _, valid_precision, valid_recall = model.test(os.path.join(FASTTEXT_DATA_DIR, "valid.txt"))
    print("Validation precision: ", valid_precision)
    print("Validation recall: ", valid_recall)
    valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall)
    print("Validation F1: ", valid_f1)

    predictions = predict(model_path, os.path.join(FASTTEXT_DATA_DIR, "test.txt"))
    _predictions_to_jsonl(predictions, args.pred)
    print("Saved test predictions in", args.pred)


if __name__ == "__main__":
    main()
