from copy import deepcopy
from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField


@Predictor.register('xstance_predictor')
class XStancePredictor(Predictor):
    """
    Predictor for any model that takes in a sentence and returns
    a single class for it.  In particular, it can be used with
    the :class:`~allennlp.models.basic_classifier.BasicClassifier` model
    """
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"label"`` to the output.
        """
        question = json_dict["question"]
        comment = json_dict["comment"]
        return self._dataset_reader.text_to_instance(question, comment)

    @overrides
    def predictions_to_labeled_instances(self,
                                         instance: Instance,
                                         outputs: Dict[str, numpy.ndarray]) -> List[Instance]:
        new_instance = deepcopy(instance)
        if "probs" in outputs:
            label = numpy.argmax(outputs['probs'])
            new_instance.add_field('prediction', LabelField(int(label)))
        elif "prediction" in outputs:
            label = outputs["score"]
            new_instance.add_field('prediction', LabelField(int(label), skip_indexing=True))
        else:
            raise ValueError("probs or score not found in prediction outputs")
        return [new_instance]
