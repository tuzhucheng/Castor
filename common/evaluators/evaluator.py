from allennlp.modules.elmo import Elmo, batch_to_ids
import nltk
nltk.download('punkt')
import torch

class Evaluator(object):
    """
    Evaluates a model on a Dataset, using metrics specific to the Dataset.
    """

    def __init__(self, dataset_cls, model, embedding, data_loader, batch_size, device, keep_results=False, use_elmo=False):
        self.dataset_cls = dataset_cls
        self.model = model
        self.embedding = embedding
        self.data_loader = data_loader
        self.batch_size = batch_size
        self.device = device
        self.keep_results = keep_results
        self.elmo = None
        if use_elmo:
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)

    def get_sentence_embeddings(self, batch):
        sent1 = self.embedding(batch.sentence_1).transpose(1, 2)
        sent2 = self.embedding(batch.sentence_2).transpose(1, 2)

        if self.elmo:
            sentences1_elmo = batch.sentence_1_raw
            sentences2_elmo = batch.sentence_2_raw

            sents = [sent1, sent2]

            for idx, sentences in enumerate([sentences1_elmo, sentences2_elmo]):
                sentences = [nltk.word_tokenize(s) for s in sentences]
                character_ids = batch_to_ids(sentences)
                elmo_embeddings = self.elmo(character_ids)
                for elmo_layer in elmo_embeddings['elmo_representations']:
                    sent_t = elmo_layer.transpose(1, 2)
                    print(elmo_layer.size())
                    sent_t = sent_t.to(self.device)
                    min_size = min(sent_t.size()[2], sents[idx].size()[2])
                    sent_t = sent_t[:, :, :min_size]
                    sents[idx] = sents[idx][:, :, :min_size]
                    sents[idx] = torch.cat((sents[idx], sent_t), 1)

            sent1 = sents[0]
            sent2 = sents[1]
            print()

        return sent1, sent2

    def get_scores(self):
        """
        Get the scores used to evaluate the model.
        Should return ([score1, score2, ..], [score1_name, score2_name, ...]).
        The first score is the primary score used to determine if the model has improved.
        """
        raise NotImplementedError('Evaluator subclass needs to implement get_score')
