from typing import List, Dict

from overrides import overrides
import numpy

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.data.fields import LabelField, ArrayField
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("two_outputs")
class TwoOutputsTextClassifierPredictor(Predictor):
    """
    Predictor for the model with two outputs (./two_outputs_clf.py)
    Registered as a `Predictor` with name "two_outputs".

    Allows for the encoder pooled text representation outputs.
    """

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        Runs the underlying model, and adds the `"label"` to the output.
        """
        sentence = json_dict["sentence"]
        reader_has_tokenizer = (
                getattr(self._dataset_reader, "tokenizer", None) is not None
                or getattr(self._dataset_reader, "_tokenizer", None) is not None
        )
        if not reader_has_tokenizer:
            tokenizer = SpacyTokenizer()
            sentence = tokenizer.tokenize(sentence)
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def predictions_to_labeled_instances(
            self, instance: Instance, outputs: Dict[str, numpy.ndarray]
    ) -> List[Instance]:
        new_instance = instance.duplicate()
        label1 = numpy.argmax(outputs["probs1"])
        new_instance.add_field("label1", LabelField(int(label1), skip_indexing=True))
        label2 = numpy.argmax(outputs["probs2"])
        new_instance.add_field("label2", LabelField(int(label2), skip_indexing=True))

        new_instance.add_field("text_repr", ArrayField(outputs["text_repr"]))
        return [new_instance]
