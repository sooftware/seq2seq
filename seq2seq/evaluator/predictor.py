import torch


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def get_decoder_features(self, src_seq):
        src_id_seq = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1, -1)
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()

        with torch.no_grad():
            output, ret_dict = self.model(src_id_seq, torch.Tensor([len(src_seq)]), teacher_forcing_ratio=0.0)

        return ret_dict

    def predict(self, src_seq):
        """ Make prediction given `src_seq` as input.
        Args:
            src_seq (list): list of tokens in source language
        Returns:
            tgt_seq (list): list of tokens in target language as predicted
            by the pre-trained model
        """
        ret_dict = self.get_decoder_features(src_seq)
        length = ret_dict['length'][0]

        tgt_id_seq = [ret_dict['sequence_symbol'][di][0].data[0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
        return tgt_seq
