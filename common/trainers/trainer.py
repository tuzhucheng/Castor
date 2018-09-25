from allennlp.modules.elmo import Elmo, batch_to_ids
import nltk
nltk.download('punkt')
import torch
class Trainer(object):

    """
    Abstraction for training a model on a Dataset.
    """

    def __init__(self, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator=None, device=None, use_elmo=False):
        self.model = model
        self.embedding = embedding
        self.optimizer = trainer_config.get('optimizer')
        self.train_loader = train_loader
        self.batch_size = trainer_config.get('batch_size')
        self.log_interval = trainer_config.get('log_interval')
        self.dev_log_interval = trainer_config.get('dev_log_interval')
        self.model_outfile = trainer_config.get('model_outfile')
        self.lr_reduce_factor = trainer_config.get('lr_reduce_factor')
        self.patience = trainer_config.get('patience')
        self.use_tensorboard = trainer_config.get('tensorboard')
        self.clip_norm = trainer_config.get('clip_norm')
        self.elmo = None
        self.device = device
        if use_elmo:
            if not device:
                raise Exception('device must be specified for the use of ELMo')
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)

        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(log_dir=None, comment='' if trainer_config['run_label'] is None else trainer_config['run_label'])
        self.logger = trainer_config.get('logger')

        self.train_evaluator = train_evaluator
        self.test_evaluator = test_evaluator
        self.dev_evaluator = dev_evaluator

    def evaluate(self, evaluator, dataset_name):
        scores, metric_names = evaluator.get_scores()
        if self.logger is not None:
            self.logger.info('Evaluation metrics for {}:'.format(dataset_name))
            self.logger.info('\t'.join([' '] + metric_names))
            self.logger.info('\t'.join([dataset_name] + list(map(str, scores))))
        return scores

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
                    sent_t = sent_t.to(self.device)
                    min_size = min(sent_t.size()[2], sents[idx].size()[2])
                    sent_t = sent_t[:, :, :min_size]
                    sents[idx] = sents[idx][:, :, :min_size]
                    sents[idx] = torch.cat((sents[idx], sent_t), 1)

            sent1 = sents[0]
            sent2 = sents[1]

        return sent1, sent2

    def train_epoch(self, epoch):
        raise NotImplementedError()

    def train(self, epochs):
        raise NotImplementedError()
