import logging
import pickle
from typing import Dict, Iterable, Optional, Union

import torch
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers import TextClassificationJsonReader
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (
    TextField,
    Field,
    LabelField,
    MetadataField
)
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from treelstm import calculate_evaluation_orders

logger = logging.getLogger(__name__)


def _label_node_index(node, n=0):
    node['index'] = n
    for child in node['children']:
        n += 1
        _label_node_index(child, n)


def _gather_node_attributes(node, key):
    features = [node[key]]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list


def convert_tree_to_tensors(tree, device=torch.device('cuda')):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    # print(tree)

    # There is no way to represent a single EDU sentence as a tree than to double this EDU as a root-child pair
    # Warning! By default EDUs are filtered in _read function!
    if not tree['children']:
        tree['children'] = [tree.copy()]

    _label_node_index(tree)

    spans = _gather_node_attributes(tree, 'spans')
    rel_labels = _gather_node_attributes(tree, 'rel_labels')
    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(spans))

    return {
        'spans': spans,
        'rel_labels': torch.tensor(rel_labels, device=device, dtype=torch.int64),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }


def match_offsets(offsets, tokens, service_tokens=True):
    # Recounts tokens offsets to subwords offsets
    current_offset = 0
    j = -1
    matching_result = []
    for i, tok in enumerate(tokens):
        if current_offset >= len(offsets):
            break

        # Service tokens, such as [CLS], do not have offsets, so skipping them
        if tok.idx and tok.idx > offsets[current_offset][0]:
            if j == -1:
                j = i - 1
            if tok.idx_end > offsets[current_offset][1]:
                matching_result.append([j, i])
                j = i
                current_offset += 1

    if len(matching_result) < len(offsets):
        if service_tokens:
            matching_result.append([j, i])
        else:
            matching_result.append([j, i + 1])

    return matching_result


def collect_discourse_info(tree, text, tokens, relation='root'):
    # {'offsets': [0, 12]),
    #  'labels': [1],
    #  'children': [{'offsets': [0, 6]),
    #    'labels': [0],
    #    'children': []},
    #   {'offsets': [6, 12]),
    #    'labels': [0],
    #    'children': [...

    high_level_relations = {
        'coherence': ['background', 'elaboration', 'restatement', 'interpretation-evaluation', 'preparation',
                      'solutionhood'],
        'causal-argumentative:contrastive': ['concession', 'contrast', 'comparison'],
        'causal-argumentative:causal': ['purpose', 'evidence', 'cause-effect'],
        'causal-argumentative:condition': ['condition'],
        'structural': ['sequence', 'joint', 'same-unit'],
        'attribution': ['attribution']
    }

    if relation != 'root':
        rel, position = relation.split('_')
        for section, low_level_rels in high_level_relations.items():
            if rel in low_level_rels:
                relation = section + '_' + position

    relations = ['coherence_N', 'coherence_S',
                 'causal-argumentative:contrastive_N', 'causal-argumentative:contrastive_S',
                 'causal-argumentative:causal_N', 'causal-argumentative:causal_S',
                 'causal-argumentative:condition_N', 'causal-argumentative:condition_S',
                 'structural_N',
                 'attribution_N', 'attribution_S',
                 'root']

    # relations = ['attribution_N', 'attribution_S',
    #              'background_N', 'background_S',
    #              'cause-effect_N', 'cause-effect_S',
    #              'comparison_N',
    #              'concession_N', 'concession_S',
    #              'condition_N', 'condition_S',
    #              'contrast_N',
    #              'elaboration_N', 'elaboration_S',
    #              'evidence_N', 'evidence_S',
    #              'interpretation-evaluation_N', 'interpretation-evaluation_S',
    #              'joint_N',
    #              'preparation_S', 'preparation_N',
    #              'purpose_N', 'purpose_S',
    #              'restatement_N',
    #              'same-unit_N',
    #              'sequence_N',
    #              'solutionhood_S', 'solutionhood_N',
    #              'root']
    relation_dict = {relations[i]: i for i in range(len(relations))}

    result = dict()
    start = text.find(tree.text)
    end = start + len(tree.text)
    result['spans'] = match_offsets(offsets=[[start, end]], tokens=tokens)
    result['rel_labels'] = [relation_dict.get(relation, relation_dict.get('structural_N'))]
    if tree.relation == 'elementary':
        result['children'] = []
    else:
        left_suffix = 'S' if tree.nuclearity == 'SN' else 'N'
        right_suffix = 'S' if tree.nuclearity == 'NS' else 'N'
        result['children'] = [
            collect_discourse_info(tree.left, text, tokens, relation=tree.relation + '_' + left_suffix),
            collect_discourse_info(tree.right, text, tokens, relation=tree.relation + '_' + right_suffix)]

    return result


@DatasetReader.register("rst_trees")
class IsaNLPRSTDatasetReader(TextClassificationJsonReader):
    """
    Reads RST parses from IsaNLP library discourse module.

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        We use this to define the input representation for the text.  See :class:`TokenIndexer`.
        Note that the `output` tags will always correspond to single token IDs based on how they
        are pre-tokenised in the data file.
    rst_label_namespace : `str`, optional, (default = "rst")
        The RST relation tag namespace
    """

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 segment_sentences: bool = False,
                 rst_label_namespace: str = "rst",
                 max_sequence_length: int = None,
                 skip_label_indexing: bool = False,
                 num_labels1: Optional[int] = None,
                 num_labels2: Optional[int] = None,
                 **kwargs,
                 ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            token_indexers=token_indexers,
            segment_sentences=segment_sentences,
            max_sequence_length=max_sequence_length,
            skip_label_indexing=skip_label_indexing,
            **kwargs,
        )
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._rst_label_namespace = rst_label_namespace
        self._num_labels1 = num_labels1
        self._num_labels2 = num_labels2

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        File with annotations is a pickle file with pandas dataframe containing keys ['tree', 'label1', 'label2']
        """

        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        data = pickle.load(open(file_path, 'rb'))
        is_not_edu = data.tree.map(lambda x: x.relation != 'elementary')  # No need in EDUs
        data = data[is_not_edu]

        for idx, row in data.iterrows():
            tree, label1, label2 = row.tree, row.label1, row.label2
            text = tree.text.lower()
            labels1 = self.correct_labels(label1)
            labels2 = self.correct_labels(label2)

            instance = self.text_to_instance(text=text, tree=tree,
                                             labels1=labels1, labels2=labels2)
            if instance is not None:
                # print(instance)
                yield instance

    def correct_labels(self, labels):
        if labels is not None:
            if self._skip_label_indexing:
                try:
                    labels = [int(label) for label in labels]
                except ValueError:
                    raise ValueError(
                        "Labels must be integers if skip_label_indexing is True."
                    )
            else:
                labels = [str(label) for label in labels.split()][0]
        return labels

    @overrides
    def text_to_instance(
            self,
            text: str,
            tree=None,
            labels1: Union[str, int] = None,
            labels2: Union[str, int] = None,
    ) -> Instance:  # type: ignore
        """
        # Parameters
        text : `str`, required.
            The text to classify
        tree : `isanlp.annotation_rst.DiscourseUnit`, required.
            The RST tree for input text
        labels : `List[Union[str, int]]`, optional, (default = `None`).
            The labels for this text.
        # Returns
        An `Instance` containing the following fields:
            - tokens (`TextField`) :
              The tokens in the sentence or phrase.
            - rst_tree (`MetadataField`) :
              RST tree in a form of python dictionary
            - label (`LabelField`) :
              The labels of the sentence or phrase.
        """

        fields: Dict[str, Field] = {}

        tokens = self._tokenizer.tokenize(text)
        if self._max_sequence_length is not None:
            tokens = self._truncate(tokens)
        fields["tokens"] = TextField(tokens, self._token_indexers)

        rst_info = collect_discourse_info(tree, text, tokens)
        tensor_rst_info = convert_tree_to_tensors(rst_info)
        fields["metadata"] = MetadataField(tensor_rst_info)

        if labels1 is not None:
            fields["label1"] = LabelField(
                labels1, skip_indexing=self._skip_label_indexing,
                label_namespace='labels1',
            )
        if labels2 is not None:
            fields["label2"] = LabelField(
                labels2, skip_indexing=self._skip_label_indexing,
                label_namespace='labels2',
            )

        return Instance(fields)
