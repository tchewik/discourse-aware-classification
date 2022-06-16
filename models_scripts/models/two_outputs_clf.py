from typing import Dict, Optional

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides
import numpy as np


@Model.register("two_outputs")
class TwoOutputsTextClassifier(Model):
    """
    This `Model` implements a weighted text classifier with two outputs. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    FIVE classification layers, which project into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "two_outputs".

    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    seq2seq_encoder : `Seq2SeqEncoder`, optional (default=`None`)
        Optional Seq2Seq encoder layer for the input text.
    seq2vec_encoder : `Seq2VecEncoder`
        Required Seq2Vec encoder layer. If `seq2seq_encoder` is provided, this encoder
        will pool its output. Otherwise, this encoder will operate directly on the output
        of the `text_field_embedder`.
    feedforward : `FeedForward`, optional, (default = `None`)
        An optional feedforward layer to apply after the seq2vec_encoder.
    dropout : `float`, optional (default = `None`)
        Dropout percentage to use.
    num_labels1 : `int`, optional (default = `None`)
    num_labels2 : `int`, optional (default = `None`)
        Number of labels to project to in classification layers. By default, the classification layers will
        project to the size of the vocabulary namespace corresponding to labels.
    namespace : `str`, optional (default = `"tokens"`)
        Vocabulary namespace corresponding to the input text. By default, we use the "tokens" namespace.
    label_namespace1 : `str`, optional (default = `"labels1"`)
    label_namespace2 : `str`, optional (default = `"labels2"`)
        Vocabulary namespace corresponding to labels of each level. By default, we use the "labels1 ... labels5" namespace.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    class_weights : 'list', deprecated; actual class weights for outputs 1..5 are detalized in the __init__()
    """

    def __init__(
            self,
            vocab: Vocabulary,
            text_field_embedder: TextFieldEmbedder,
            seq2vec_encoder: Seq2VecEncoder,
            seq2seq_encoder: Seq2SeqEncoder = None,
            feedforward: Optional[FeedForward] = None,
            dropout: float = None,
            num_labels1: int = None,
            num_labels2: int = None,
            label_namespace1: str = "labels1",
            label_namespace2: str = "labels2",
            namespace: str = "tokens",
            initializer: InitializerApplicator = InitializerApplicator(),
            label1_weights: list = [1., 1., 1., 1.],
            label2_weights: list = [1., 1., 1., 1.],
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder
        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._seq2vec_encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._label_namespace1 = label_namespace1
        self._label_namespace2 = label_namespace2
        self._namespace = namespace

        if num_labels1:
            self._num_labels1 = num_labels1
            self._num_labels2 = num_labels2
        else:
            raise BaseException('Specify number of labels for output 1 and output 2')

        self._classification_layer1 = torch.nn.Linear(self._classifier_input_dim, self._num_labels1)
        self._classification_layer2 = torch.nn.Linear(self._classifier_input_dim, self._num_labels2)

        self.metrics = {"accuracy_1": CategoricalAccuracy(),
                        "accuracy_2": CategoricalAccuracy(),
                        "f1_1_1": F1Measure(1),
                        "f1_1_2": F1Measure(2),
                        "f1_1_3": F1Measure(3),
                        "f1_2_1": F1Measure(1),
                        "f1_2_2": F1Measure(2),
                        "f1_2_3": F1Measure(3)}

        self._loss1 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(label1_weights))
        self._loss2 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(label2_weights))

        initializer(self)

    def forward(  # type: ignore
            self,
            tokens: TextFieldTensors,
            label1: torch.IntTensor = None,
            label2: torch.IntTensor = None,
            metadata: MetadataField = None,
            *args, **kwargs
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`
            From a `TextField`
        label1 : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`
        label2 : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`

        # Returns

        An output dictionary consisting of:

            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs1` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the first level label.
            - `probs2` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the second level label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        embedded_text = self._text_field_embedder(tokens)
        mask = get_text_field_mask(tokens)

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        embedded_text = self._seq2vec_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        logits1 = self._classification_layer1(embedded_text)
        probs1 = torch.nn.functional.softmax(logits1, dim=-1)
        logits2 = self._classification_layer2(embedded_text)
        probs2 = torch.nn.functional.softmax(logits2, dim=-1)

        output_dict = {"logits1": logits1, "probs1": probs1,
                       "logits2": logits2, "probs2": probs2}

        output_dict["token_ids"] = util.get_token_ids_from_text_field_tensors(tokens)

        losses = []
        if label1 is not None:
            losses.append(self._loss1(logits1, label1.long()))
            self.metrics['accuracy_1'](logits1, label1)
            self.metrics['f1_1_1'](logits1, label1)
            self.metrics['f1_1_2'](logits1, label1)
            self.metrics['f1_1_3'](logits1, label1)

        if label2 is not None:
            losses.append(self._loss2(logits2, label2.long()))
            self.metrics['accuracy_2'](logits2, label2)
            self.metrics['f1_2_1'](logits2, label2)
            self.metrics['f1_2_2'](logits2, label2)
            self.metrics['f1_2_3'](logits2, label2)

        if losses:
            loss = losses[0] + losses[1]
            output_dict["loss"] = loss
        else:
            output_dict["text_repr"] = embedded_text

        return output_dict

    def _human_readable(self, predictions, lvl_label_namespace):
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(lvl_label_namespace).get(label_idx, str(label_idx))
            classes.append(label_str)

        return classes

    @overrides
    def make_output_human_readable(
            self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """

        output_dict["label1"] = self._human_readable(output_dict["probs1"], self._label_namespace1)
        output_dict["label2"] = self._human_readable(output_dict["probs2"], self._label_namespace2)

        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(token_id.item(), namespace=self._namespace)
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens
        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {f"accuracy_{i}": self.metrics[f"accuracy_{i}"].get_metric(reset=reset) for i in [1, 2]}

        metrics.update({f'f1_1_{i}': self.metrics[f"f1_1_{i}"].get_metric(reset=reset)['f1'] for i in range(1, 4)})
        metrics['f1_1_mean'] = np.mean([metrics[f'f1_1_{i}'] for i in range(1, 4)])

        metrics.update({f'f1_2_{i}': self.metrics[f"f1_2_{i}"].get_metric(reset=reset)['f1'] for i in range(1, 4)})
        metrics['f1_2_mean'] = np.mean([metrics[f'f1_2_{i}'] for i in range(1, 4)])

        metrics['all_mean'] = np.mean([metrics['f1_1_mean'], metrics['f1_2_mean']])

        return metrics

    default_predictor = "two_outputs"
