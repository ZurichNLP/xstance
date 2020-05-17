local bert_model = "bert-base-multilingual-cased";

{
    "dataset_reader": {
        "lazy": false,
        "type": "xstance_reader",
        "max_sequence_length": 512,
        "skip_label_indexing": false,
        "tokenizer": {
            "type": "pretrained_transformer",
            "do_lowercase": false,
            "model_name": bert_model
        },
        "token_indexers": {
            "bert": {
                "type": "bert-pretrained",
                "do_lowercase": false,
                "pretrained_model": bert_model
            }
        }
    },
    "train_data_path": "../data/train.jsonl",
    "validation_data_path": "../data/valid.jsonl",
    "model": {
        "type": "bert_for_classification",
        "bert_model": bert_model,
        "dropout": 0.1
    },
    "iterator": {
        "type": "basic",
        "batch_size": 16,
    },
    "trainer": {
        "optimizer": {
            "type": "bert_adam",
            "warmup": 0.1,
            "t_total": 8580,
            "lr": 0.00002
        },
        "validation_metric": "+accuracy",
        "num_serialized_models_to_keep": 1,
        "num_epochs": 3,
        "patience": 5,
        "cuda_device": 0
    }
}
