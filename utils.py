from typing import Dict, List, Set, Tuple

# import nltk
import pandas as pd
import torch
from torchcrf import CRF as CRFDecoder
from tqdm import tqdm
from wasabi import msg

from spacy.training import offsets_to_biluo_tags

from data_preparation import get_sentence_docs
from postprocessing_utils import postprocessing

# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)


# For feature encodings. We have a special UNKNOWN feature
UNK_SYMBOL = "<UNK>"
# For padding our variable length batches
PAD_SYMBOL = "<PAD>"

# def json_to_dataframe(data):
#     id = []
#     text = []
#     start = []
#     end = []
#     entity = []
#     label = []

#     all_labels = set()

#     for d in data:
#         id_text = d['id']
#         data = d['data']
#         for x in d['annotations']:
#             info = x['result']
#             for i in info:
#                 i_dict = i['value']
#                 id.append(id_text)
#                 text.append(data['text'])
#                 start.append(i_dict['start'])
#                 end.append(i_dict['end'])
#                 entity.append(i_dict['text'])
#                 label.append(i_dict['labels'][0])
#                 all_labels.add(i_dict['labels'][0])
            
#     df_train = pd.DataFrame(list(zip(id,text,start,end,entity,label)), columns=['id','text','start','end','entity','label'])
#     return (df_train, list(all_labels))


# def get_spacy_texts_from_json(data, nlp_preamble_splitting):
#     ids = []
#     texts = []

#     for d in data:
#         ids.append(d['id'])
#         text = d['data']['text']
#         texts.append(nlp_preamble_splitting(text))
    
#     df_texts = pd.DataFrame(list(zip(ids,texts)), columns=['id','text'])

#     return df_texts


def predict_baseline(json_data, model, nlp_preamble_splitting, text_type, do_postprocess):
    # returns the prediction of all entities in each text
    result = []

    for data in json_data:
        doc = nlp_preamble_splitting(data['text'])

        if text_type=='doc':
            predicted=model(doc)
        else:
            predicted=get_sentence_docs(doc, model)
        
        try:
            if do_postprocess:
                predicted=postprocessing(predicted)
        except:
            msg.warn('There was some issue while performing postprocessing, skipping postprocessing...')

        predicted_json = predicted.to_json()
        
        result.append({
            'text': predicted_json['text'],
            'ents': predicted_json['ents']
        })

    return result


def prepare_data_from_raw_json(json_data):
    result = []

    for data in json_data:
        out = {}
        offset_entities_list = []

        for annot in data['annotations']:
            entities = annot['result']
            for entity in entities:
                offset_entities_list.append({
                    'start': entity['value']['start'],
                    'end': entity['value']['end'],
                    'label': entity['value']['labels'][0]
                })
        
        out['text'] = data['data']['text']
        out['ents'] = offset_entities_list
        result.append(out)
    
    return result


def get_iob_tags(text, results_data):
    def get_offset_list_from_json_data(results_data):
        entities = []

        for entity in results_data:
            entities.append((entity['start'], entity['end'], entity['label']))
        return entities

    entity_offset_list = get_offset_list_from_json_data(results_data)
    biluo_tags = offsets_to_biluo_tags(text, entity_offset_list, missing="O")

    def convert_biluo_to_iob(tags):
        iob_tags = []
        is_misaligned = False
        for tag in tags:
            if tag == '-':
                is_misaligned = True
            
            tag_split = tag.split("-")

            if tag_split[0] == 'L':
                tag = "I-" + tag_split[1]
            elif tag_split[0] == 'U':
                tag = "B-" + tag_split[1]
            
            iob_tags.append(tag)
        if is_misaligned:
            print(text, entity_offset_list, tags)
        return iob_tags

    return convert_biluo_to_iob(biluo_tags)


def load_processed_json_data(
    json_data,
    nlp_preamble_splitting
) -> Tuple[List[str], List[str]]:
    """Load the training data, producing a List of sentences, each comprising
    a List of tokens, and a List of their corresponding NER tags

    Args:
        json_data: JSON data
        nlp_preamble_splitting: spacy Model

    Returns:
       Tuple[List[str], List[str]]: The tokens and tags
    """
    token_sents = []
    tag_sents = []

    for json in json_data:
        # Append list of tokens in text
        text = json['text']
        text = nlp_preamble_splitting(text)
        token_sents.append([word.text for word in text])
        
        # Create IOB Tagging
        tag_sents.append(get_iob_tags(text, json['ents']))

    return token_sents, tag_sents


def get_all_labels(training_tag_data: List[List[str]]):
    all_labels = set()

    for data in training_tag_data: 
        all_labels.update(data)
    
    all_labels = sorted(list(all_labels))
    
    return all_labels


def make_labels2i(all_labels: List[str]):
    # Initialize labels2i with the PAD_SYMBOL.
    labels2i = {PAD_SYMBOL: 0}
    i = 1

    for label in all_labels:
        labels2i[label] = i
        i += 1
    
    return labels2i


class NERTagger(torch.nn.Module):
    """NER tagger in pytorch, 
    relying on (pytorch-crf)[https://pytorch-crf.readthedocs.io/en/stable/]
    """
    def __init__(self, features_dim: int, num_tags: int):
        """
        Args:
            num_tags (int): The number of NER tags.
            features_dim (int): Dimension of each feature vector.
                This should correspond to the number of possible features.
        """
        super().__init__()
        self.num_tags = num_tags
        self.features_dim = features_dim
        # The matrix for computing emission probabilities before
        # incorporating the label sequence in the CRFDecoder.
        # NOTE: We use nn.Embedding for its sparse implementation,
        #   but this is not actually getting embeddings in the general sense
        #   of learning representations. If you are interested in this detail, please ask!
        self.emissions_scorer = torch.nn.Embedding(features_dim, num_tags)
        # Initialize the CRF using pytorch-crf. This adds the structured prediction function 
        # on top of our scorer. Note that the scorer works essentially like a logistic regression classifier
        # But we add the CRF on top.
        self.crf_decoder = CRFDecoder(self.num_tags, batch_first=True)

    def forward(
        self, input_seq: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the log likelihood of the gold tags. This assumes
        we are in training mode.

        Args:
            input_seq (torch.Tensor): sequence_length x batch_size x num_features tensor
                of the featurized input sentence.
            tags (torch.Tensor): sequence_length x batch_size tensor of the tags.
            mask (torch.Tensor): THe mask marking which tokens are dummy pads.

        Returns:
             torch.Tensor: A batch_size x 1 tensor of the (float) sum of log_likelihoods
                for each tag.
        """
        # Get the probability of each label for each token.
        emissions = self.make_emissions(input_seq)
        # Compute the log likelihood of the gold tags 
        # given the input. We get y_i and y_{i-1} from the gold tags.
        return self.crf_decoder(emissions, tags, mask=mask, reduction="mean")

    def decode(self, input_seq: torch.Tensor) -> List[int]:
        emissions = self.make_emissions(input_seq)
        # Decode the argmax sequence of labels with viterbi
        return self.crf_decoder.decode(emissions)

    def make_emissions(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Compute a probability distribution over the tags for each
        input word.

        Args:
            input_seq (torch.Tensor): batch_size x sequence_length x num_features tensor
                of the featurized input sentence.
        """
        # -> batch_size x seq_len x num_tags
        # This gives a distribution over the tags for each token
        # in the input sequence
        # We sum over the 2nd to last dim (#features) because we use embeddings as a hack
        # to get efficient sparse implementation.
        # Embedding each feature and summing is equivalent to the matrix-multiply
        # of a single layer.
        return torch.sum(self.emissions_scorer(input_seq), dim=-2)


def make_features_dict(all_features: Set) -> Dict:
    # Add 2 to each feature index to reserve 0 for PAD and 1 for UNK
    features_dict = {f: i+2 for i, f in enumerate(all_features)}
    features_dict[PAD_SYMBOL] = 0
    features_dict[UNK_SYMBOL] = 1
    print(f"Found {len(features_dict)} features")

    return features_dict


def encode_token_features(features: List[str], features_dict: Dict[str, int]) -> torch.Tensor:
    """Turn the features into a Tensor of integers.
    Note that we let nn.Embedding handle sparse layer, so we can pass a dense tensor of
    indices, rather than an indicator tensor.

    Args:
        features (List[str]): The string features to encode.
        features_dict (Dict[str, int]): The encoding of features to indices.

    Returns:
        torch.Tensor: The encoded features.
    """
    return torch.LongTensor([
        features_dict.get(feat, features_dict[UNK_SYMBOL]) for feat in features
    ])


def predict(model: torch.nn.Module, feature_sents: List[List[int]]) -> List[torch.Tensor]:
    """Make predictions with the input model. Return a List of tensors.

    Args:
        model (torch.nn.Module): The trained model.
        feature_sents (List[List[int]]): The feature Lists for each sentence.

    Returns:
        List[torch.Tensor]: A List of predicted integers.
    """
    out = []
    for features in feature_sents:
        # Dummy batch dimension
        features = features.unsqueeze(0)
        # -> List[List[int]]
        preds = model.decode(features)
        preds = [torch.tensor(p) for p in preds]
        out.extend(preds)

    return out


###########
# PADDING #
# We implement methods for padding inputs and targets in order to ensure that our variable
# length sequences can form matrices. This is to enable batrching.
def pad_tensor(tensor: torch.Tensor, pad_max: int, pad_idx: int) -> torch.Tensor:
    """Pad a single tensor up to `pad_max` with the `pad_idx`.

    Args:
        tensor (torch.Tensor): the input tensor to be padded.
        pad_max (int): length of the requested output tensor.
        pad_idx (int): index of the pad.

    Returns:
        torch.Tensor: _description_
    """
    padding = pad_max - len(tensor)
    return torch.nn.functional.pad(tensor, (0, padding), "constant", pad_idx)


def pad_2d_tensor(
    tensor: torch.Tensor,
    pad_max: int,
    num_features: int,
    pad_idx: int
) -> torch.Tensor:
    """Pad a tensor with a pads vector.

    Args:
        tensor (torch.Tensor): The seq_len x num_features input tensor to pad.
        pad_max (int): The requested length to pad to.
        num_features (int): The number of dimensions in the pad vectors.
        pad_idx (int): The index of the pad feature.

    Returns:
        torch.Tensor: tensor padded up to pad_max.
    """
    padding_len = pad_max - len(tensor)
    pads_matrix = torch.ones(padding_len, num_features) * pad_idx
    return torch.cat((tensor, pads_matrix.long()))


def pad_labels(labels_list: List[torch.Tensor], pad_idx: int) -> List[torch.Tensor]:
    """Pad each labels tensor in the list, so each tensor is the same size.

    For example, if we have a list of 2 tensors: [Tensor([1,2]), Tensor([1,2,3])],
        then we want to transform it to:  [Tensor([1,2, PAD]), Tensor([1,2,3])].

    Args:
        labels_list (List[torch.Tensor]): The list of label tensors in a batch.
        pad_idx (int): The index representing a pad.

    Returns:
        List[torch.Tensor]: The List of padded Tensors.
    """
    pad_max = max([len(l) for l in labels_list])
    return [pad_tensor(l, pad_max, pad_idx) for l in labels_list]


def pad_features(features_list: List[torch.Tensor], pad_idx: int) -> List[torch.Tensor]:
    """Pad each feature tensor in the list, so each tensor is the same size.

    For example, if we have a list of 2 tensors: [Tensor([1,2]), Tensor([1,2,3])],
        then we want to transform it to:  [Tensor([1,2, PAD]), Tensor([1,2,3])].

    Args:
        features_list (List[torch.Tensor]): The list of sequence_length x num_features tensors in a batch.
        pad_idx (int): The index representing a pad.

    Returns:
        List[torch.Tensor]: The List of padded Tensors.
    """
    pad_max = max([len(l) for l in features_list])
    num_features = features_list[0].size(1)
    return [pad_2d_tensor(f, pad_max, num_features, pad_idx) for f in features_list]


def build_features_set(train_features: List[List[List[str]]]) -> Set:
    """Build a set of all possible features. We limit this to only those in train.

    Args:
        train_features (List[List[List[str]]]): A List representing sentences.

    Returns:
        Set: The set of all unique features
    """
    all_features = set()

    print("Building features set!")
    for features_sent in tqdm(train_features):
        for features in features_sent:
            for f in features:
                all_features.add(f)

    return all_features


def encode_features(
    feature_sents: List[List[List[str]]], features_dict: Dict[str, int]
) -> List[torch.LongTensor]:
    """Encode the features with features_dict.

    Args:
        feature_sents (List[List[List[str]]]): The List of sentences, of feature Lists.
        features_dict (Dict[str, int]): The encoding dict.

    Returns:
        List[torch.LongTensor]: A List of Tensors with encoded features
    """
    encoded_features = []
    
    for feature_sent in feature_sents:
        encoded = [
            encode_token_features(token_features, features_dict) for token_features in feature_sent
        ]
        encoded_features.append(torch.stack(encoded))

    return encoded_features


def encode_labels(tag_sents: List[List[str]], labels2i: Dict[str, int]) -> List[torch.LongTensor]:
    """Encode the labels using label2i. This results in one Tensor for each sentence.

    Args:
        tag_sents (List[List[str]]): The tag for each token in each sentence.
        labels2i (Dict[str, int]): The encoding dict for labels.

    Returns:
        List[torch.LongTensor]: Encoded labels.
    """
    encoded_labels = []
    for labels in tag_sents:
        encoded_labels.append(torch.LongTensor([
            labels2i[l] for l in labels
        ]))

    return encoded_labels
