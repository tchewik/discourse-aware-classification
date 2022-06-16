from typing import Dict, Optional

import numpy as np
import torch
import treelstm
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from overrides import overrides


class ChildSumTreeLSTM(torch.nn.Module):
    '''PyTorch Child-Sum Tree LSTM model
    See Tai et al. 2015 https://arxiv.org/abs/1503.00075 for model description.
    '''

    def __init__(self, in_features, out_features):
        '''TreeLSTM class initializer

        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, features, node_order, adjacency_list, edge_order):
        '''Run TreeLSTM model on a tree data structure with node features

        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        '''

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order)

        return h, c

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            h_sum = torch.stack(parent_list)
            iou = self.W_iou(x) + self.U_iou(h_sum)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])


class BinaryTreeLSTM(torch.nn.Module):
    '''PyTorch N-ary Tree LSTM model
    See Tai et al. 2015 https://arxiv.org/abs/1503.00075 for model description.
    '''

    def __init__(self, in_features, out_features):
        '''TreeLSTM class initializer

        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou_left = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)
        self.U_iou_right = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained separate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f_left = torch.nn.Linear(self.out_features, self.out_features, bias=False)
        self.U_f_right = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, features, node_order, adjacency_list, edge_order):
        '''Run TreeLSTM model on a tree data structure with node features

        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        '''

        # Total number of nodes in every tree in the batch
        batch_size = node_order.shape[0]

        # Retrive device the model is currently loaded on to generate h, c, and h_sum result buffers
        device = next(self.parameters()).device

        # h and c states for every node in the batch
        h = torch.zeros(batch_size, self.out_features, device=device)
        c = torch.zeros(batch_size, self.out_features, device=device)

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order)

        return h, c

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]

            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list_left = [item[0] for item in parent_children]
            parent_list_right = [item[1] for item in parent_children]

            h_sum_left = torch.stack(parent_list_left)
            h_sum_right = torch.stack(parent_list_right)
            iou = self.W_iou(x) + self.U_iou_left(h_sum_left) + self.U_iou_right(h_sum_right)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f_left = torch.sigmoid(self.W_f(features[parent_indexes, :]) + self.U_f_left(child_h))
            f_right = torch.sigmoid(self.W_f(features[parent_indexes, :]) + self.U_f_right(child_h))

            # fc is a tensor of size e x M
            fc = (f_left + f_right) * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])


@Model.register("two_outputs_tree_lstm")
class TwoOutputsTreeLSTM(Model):
    """
    PyTorch Child-Sum Tree LSTM model

    See Tai et al. 2015 https://arxiv.org/abs/1503.00075 for model description.

    This `Model` implements a Child-Sum Tree LSTM model classifier with two outputs. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    two classification layers, which project into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.

    Registered as a `Model` with name "two_outputs_tree_lstm".

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
            treelstm_type: str = 'binary',
            dropout: float = None,
            num_labels1: int = None,
            num_labels2: int = None,
            label_namespace1: str = "labels1",
            label_namespace2: str = "labels2",
            namespace: str = "tokens",
            initializer: InitializerApplicator = InitializerApplicator(),
            label1_weights: list = [1., 1., 1., 1.],
            label2_weights: list = [1., 1., 1., 1.],
            treelstm_hidden_size: int = 256,
            rst_weights: list = None,
            **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self._label_namespace1 = label_namespace1
        self._label_namespace2 = label_namespace2
        self._namespace = namespace

        if num_labels1:
            self._num_labels1 = num_labels1
            self._num_labels2 = num_labels2
        else:
            raise BaseException('Specify number of labels for output 1 and output 2')

        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._seq2vec_encoder = seq2vec_encoder

        self._treelstm_output_dim = treelstm_hidden_size
        self._treelstm_by_sample = False  # Enforce tree exploitation sample-by-sample (development only)
        self._treelstm_model = BinaryTreeLSTM if treelstm_type == 'binary' else ChildSumTreeLSTM

        self._number_rst_rels = 12
        # self._rst_lab_hidden = 1
        # self._rst_label_encoder = torch.nn.Linear(self._number_rst_rels, self._rst_lab_hidden, bias=False)
        # if not rst_weights: rst_weights = torch.tensor([1.] * (self._number_rst_rels - 1) + [0.5], dtype=torch.float32)
        # self._rst_weights = torch.nn.Parameter(rst_weights, requires_grad=True)

        self._treelstm_encoder = self._treelstm_model(
            in_features=self._num_labels1 + self._num_labels2 + self._number_rst_rels,
            out_features=self._treelstm_output_dim)

        self._feedforward = feedforward
        if feedforward:
            self._classifier_input_dim = feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self._treelstm_output_dim

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None

        self._classification_layer1 = torch.nn.Linear(self._seq2vec_encoder.get_output_dim(), self._num_labels1)
        self._classification_layer2 = torch.nn.Linear(self._seq2vec_encoder.get_output_dim(), self._num_labels2)

        self._final_ff_layer1 = torch.nn.Linear(self._classifier_input_dim, self._num_labels1)
        self._final_ff_layer2 = torch.nn.Linear(self._classifier_input_dim, self._num_labels2)

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
        metadata : Contains additional RST information

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
        embedded_text = self._text_field_embedder(tokens)  # [batch_size, sentence_len, 768]
        mask = get_text_field_mask(tokens)

        batch_size = embedded_text.shape[0]

        if self._seq2seq_encoder:
            embedded_text = self._seq2seq_encoder(embedded_text, mask=mask)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        self.device = embedded_text.device

        # нарезать из текста фичи для дерева (контекстные эмбеддинги + отношения RST)
        spans_masks = []
        max_len = embedded_text.shape[1]
        for idx in range(batch_size):
            smask = [[span[0][0] <= i < span[0][1] for i in range(max_len)] for span in metadata[idx]['spans']]
            smask = np.vstack(smask).astype(bool)
            smask[:, 0], smask[:, -1] = True, True
            spans_masks.append(smask)

        if self._treelstm_by_sample:
            encoded_rst = []
            for idx in range(batch_size):
                spans_masks_idx = torch.tensor(spans_masks[idx], device=self.device)
                rst_labels_idx = metadata[idx]['rel_labels']
                nodes_number = rst_labels_idx.shape[0]
                text_idx = torch.stack([embedded_text[idx]] * nodes_number, axis=0)

                spans_features = self._seq2vec_encoder(text_idx, mask=spans_masks_idx)  # [nodes_number, 768]

                # a. Just concatenate the number (development only)
                # spans_features = torch.cat([spans_features, rst_labels_idx], -1)
                # b. Multiply the features by trainable rst relation weight
                spans_features = spans_features #* self._rst_weights[rst_labels_idx.T]

                h, c = self._treelstm_encoder(spans_features,
                                              metadata[idx]['node_order'],
                                              metadata[idx]['adjacency_list'],
                                              metadata[idx]['edge_order'])
                encoded_rst.append(h[0])

            encoded_rst = torch.stack(encoded_rst)

        else:
            trees_in_batch = []
            for idx in range(batch_size):
                spans_masks_idx = torch.tensor(spans_masks[idx], device=self.device)
                rst_labels_idx = metadata[idx]['rel_labels']
                nodes_number = rst_labels_idx.shape[0]
                text_idx = torch.stack([embedded_text[idx]] * nodes_number, axis=0)

                # actual text span cutting
                l = [text_idx[i][spans_masks_idx[i]] for i in range(text_idx.shape[0])]
                cutted_embeddings = torch.zeros(nodes_number, embedded_text.shape[1], embedded_text.shape[2],
                                                device=self.device)  # [nodes_number, max_len, 768]
                for i in range(len(l)):
                    cutted_embeddings[i, :l[i].shape[0], :l[i].shape[1]] = l[i]

                spans_vectors = self._seq2vec_encoder(cutted_embeddings)  # [nodes_number, 768]

                with torch.no_grad():
                    pre_prediction1 = self._classification_layer1(spans_vectors)
                    pre_prediction2 = self._classification_layer2(spans_vectors)

                # a. Just concatenate the number (development only)
                # spans_vectors = torch.cat([spans_vectors, rst_labels_idx], -1)

                # b. Multiply the features by trainable rst relation weight
                # multiplier = self._rst_weights[rst_labels_idx.T[0]].unsqueeze(1)
                # multiplier = 0.5 + torch.sigmoid(self._rst_weights[rst_labels_idx.T[0]].unsqueeze(1))
                # spans_vectors = spans_features * multiplier

                # c. Add one-hot encoded relations to the embeddings
                nodes_rst_one_hot = torch.nn.functional.one_hot(rst_labels_idx.squeeze(1),
                                                                num_classes=self._number_rst_rels).float()
                spans_vectors = torch.cat([pre_prediction1, pre_prediction2, nodes_rst_one_hot], -1)

                trees_in_batch.append({
                    'features': spans_vectors,
                    'labels': rst_labels_idx,
                    'node_order': metadata[idx]['node_order'],
                    'adjacency_list': metadata[idx]['adjacency_list'],
                    'edge_order': metadata[idx]['edge_order']
                })

            batch = treelstm.batch_tree_input(trees_in_batch)


            h, c = self._treelstm_encoder(batch['features'],
                                          batch['node_order'],
                                          batch['adjacency_list'],
                                          batch['edge_order'])

            lengths = [tree['features'].shape[0] for tree in trees_in_batch]
            ## 1a. Use tree LSTM representation of the root nodes
            encoded_rst = torch.stack(
                [h[sum(lengths[:i])] for i in range(batch_size)])#.split(self._treelstm_output_dim // 2, dim=-1)

            # if self._dropout:
            #     encoded_rst = self._dropout(encoded_rst)

            ## 2b. Find weighted sum of the lstm outputs with rst relation weights
            # encoded_rst = []
            # for i, length in enumerate(lengths):
            #     start = sum(lengths[:i])
            #     end = start + length
            #     encoded_rst.append(torch.sum(torch.mul(h[start:end], self._rst_weights[trees_in_batch[i]['labels']]), axis=0) / length)
            # encoded_rst = torch.vstack(encoded_rst)

            # randval = np.random.randint(200)
            # if not randval:
            #     print('self.rst_weights =', self._rst_weights)

        logits1 = self._final_ff_layer1(encoded_rst)
        logits2 = self._final_ff_layer2(encoded_rst)

        probs1 = torch.nn.functional.softmax(logits1, dim=-1)
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

    default_predictor = "two_outputs_rst"
