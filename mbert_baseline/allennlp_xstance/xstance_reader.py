from collections import defaultdict
from typing import Dict, Union
import logging

import jsonlines
from overrides import overrides
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("xstance_reader")
class XStanceReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 max_sequence_length: int = None,
                 skip_label_indexing: bool = False,
                 ignore_questions: bool = False,
                 ignore_comments: bool = False,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._max_sequence_length = max_sequence_length
        self._skip_label_indexing = skip_label_indexing
        self.ignore_questions = ignore_questions
        self.ignore_comments = ignore_comments
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with jsonlines.open(cached_path(file_path), "r") as f:
            for i, answer in enumerate(f):
                question = answer["question"]
                comment = answer["comment"]
                label = answer.get("label", None)
                if label is not None:
                    if self._skip_label_indexing:
                        try:
                            label = int(label)
                        except ValueError:
                            raise ValueError('Labels must be integers if skip_label_indexing is True.')
                    else:
                        label = str(label)
                instance = self.text_to_instance(question=question, comment=comment, label=label)
                if i < 4:
                    logger.debug(instance)
                if instance is not None:
                    yield instance

    def _truncate(self, n, tokens):
        """
        truncate a set of tokens using the provided sequence length
        """
        if len(tokens) > n:
            tokens = tokens[:n]
        return tokens

    @overrides
    def text_to_instance(self, question: str, comment: str, label: Union[str, int] = None) -> Instance:
        fields: Dict[str, Field] = {}
        if self.ignore_questions:
            question_tokens = []
        else:
            question_tokens = self._tokenizer.tokenize(question)
        if self.ignore_comments:
            comment_tokens = []
        else:
            comment_tokens = self._tokenizer.tokenize(comment)
            comment_tokens = comment_tokens[1:]  # Do not need [CLS] in second segment
        if self._max_sequence_length is not None:
            question_tokens = self._truncate(self._max_sequence_length - len(comment_tokens) - 2, question_tokens)
        tokens = question_tokens + comment_tokens
        fields['tokens'] = TextField(tokens, self._token_indexers, )
        if label is not None:
            fields['label'] = LabelField(label,
                                         skip_indexing=self._skip_label_indexing)
        return Instance(fields)
