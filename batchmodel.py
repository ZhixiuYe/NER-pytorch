import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

START_TAG = '<START>'
STOP_TAG = '<STOP>'
EMBEDDING_DIM = 5
HIDDEN_DIM = 4
learning_rate = 0.01
weight_decay = 1e-4


def to_scalar(var):
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# compute the log sum exp of a
def log_sum_exp(vec):
    # vec 2D: 1 * tagset_size
    max_score = vec[0, argmax(vec)]
    # print 'vec size: ', vec.size()
    # print 'max_score_size: ', max_score
    # can not broadcast automatically?
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class batch_BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, char_lstm_dim=25, # batch_size=8,
                 char_to_ix=None, pre_word_embeds=None, char_embedding_dim=25, use_gpu=False,
                 char_bi_direction=True):
        super(batch_BiLSTM_CRF, self).__init__()
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        # self.batch_size = batch_size
        self.tagset_size = len(tag_to_ix)
        if char_embedding_dim is not None:
            self.char_lstm_dim = char_lstm_dim
            self.char_embeds = nn.Embedding(len(char_to_ix), char_embedding_dim)
            self.char_dropout = nn.Dropout(0.5)
            self.char_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim)
            if char_bi_direction:
                self.char_rev_lstm = nn.LSTM(char_embedding_dim, char_lstm_dim)
        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embeds is not None:
            self.use_pre_word_embeds = True
            self.word_embdes = nn.Parameter(torch.FloatTensor(pre_word_embeds))
        else:
            self.use_pre_word_embeds = False
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.dropout = nn.Dropout(0.5)
        # self.lstm = nn.LSTM(embedding_dim+char_embedding_dim*2, hidden_dim // 2,
        #                     num_layers=1, bidirectional=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        # trans is also a score tensor, not a probability
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))
        # self.transitions = nn.Parameter(
        #     torch.cat((torch.randn(3, 5), torch.Tensor(1, 5).fill_(-10000.), torch.randn(1, 5)), 0))
        # self.hidden = self.init_hidden(self.batch_size)
        self.char_lstm_hidden = self.init_char_lstm_hidden()

    def init_hidden(self, batch):
        if self.use_gpu:
            return (autograd.Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda(),
                    autograd.Variable(torch.randn(2, batch, self.hidden_dim // 2)).cuda())
        else:
            return (autograd.Variable(torch.randn(2, batch, self.hidden_dim // 2)),
                    autograd.Variable(torch.randn(2, batch, self.hidden_dim // 2)))

    def init_char_lstm_hidden(self):
        if self.use_gpu:
            return (autograd.Variable(torch.randn(1, 1, self.char_lstm_dim)).cuda(),
                    autograd.Variable(torch.randn(1, 1, self.char_lstm_dim)).cuda())
        else:
            return (autograd.Variable(torch.randn(1, 1, self.char_lstm_dim)),
                    autograd.Variable(torch.randn(1, 1, self.char_lstm_dim)))

    def _score_sentence(self, feats, tags):
        # tags is ground_truth, a list of ints, length is len(sentence)
        # feats is a 2D tensor, len(sentence) * tagset_size
        score = autograd.Variable(torch.Tensor([0]))
        if self.use_gpu:
            score = score.cuda()
            tags = torch.cat([torch.cuda.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        else:
            tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            # trans score + state score
            score += self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score += self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _get_lstm_features(self, batch_sentence, sentence_length, batch_chars2, chars2_length):

        # batch_sentence: batch_size * max_length
        # initialize lstm hidden state, h and c
        self.max_length = max(sentence_length)
        self.batch_size = len(sentence_length)
        # print(self.batch_size)
        self.hidden = self.init_hidden(self.batch_size)

        if self.use_pre_word_embeds:
            # print('use_pre:', self.use_pre_word_embeds)
            # embeds = torch.FloatTensor(self.batch_size, self.max_length, self.embedding_dim)
            for i, s in enumerate(batch_sentence):
                for j, w in enumerate(s):
                    # embeds[i, j, :] = self.word_embeds[w.data[0]]
                    if j == 0:
                        embeds1 = self.word_embdes[w.data[0]].view(1, -1)
                    else:
                        embeds1 = torch.cat((embeds1, self.word_embdes[w.data[0]].view(1, -1)), 0)
                if i == 0:
                    embeds2 = embeds1.view(1, self.max_length, self.embedding_dim)
                else:
                    embeds2 = torch.cat((embeds2, embeds1.view(1, self.max_length, self.embedding_dim)), 0)
            embeds = embeds2.transpose(0, 1)
            # embeds = embeds.view(batch_sentence.size(0), batch_sentence.size(1), -1)
        else:
            embeds = self.word_embeds(batch_sentence).transpose(0, 1)
        # print embeds.size()

        # embeds = torch.cat((embeds, char_lstm_for_embedding, char_lstm_rev_embedding), 2)

        # embeds: max_length * batch_size * embedded_size
        embeds = self.dropout(embeds)


        # convey a 3D tensor to lstm, and return a 3D tensor, its shape is the same as embeds
        # this is because that for lstm, every input product a output

        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_length)

        lstm_out, self.hidden = self.lstm(packed, self.hidden)

        # outputs: max_length * batch_size * embedded_size
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)

        assert outputs.size() == (self.max_length, self.batch_size, self.hidden_dim)
        # hidden2tag is a transformation from self.hidden_dim to self.tagset_size
        # lstm_feats: len(sentence) * batch * self.tagset_size

        lstm_feats = []
        for i, o in enumerate(outputs.transpose(0, 1)):
            lstm_feats.append(self.hidden2tag(o).view(1, self.max_length, self.tagset_size))
        lstm_feats = torch.cat(lstm_feats, 0)
        assert lstm_feats.size() == (self.batch_size, self.max_length, self.tagset_size)
        # there is no softmax
        return lstm_feats

    def _forward_alg(self, feats):
        # calculate in log domain
        # feats is len(sentence) * tagset_size
        # initialize alpha with a Tensor with values all equal to -10000.
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        # set START_TAG to max
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        # transform to a variable
        forward_var = autograd.Variable(init_alphas)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        # print 'feats: ', feats
        for feat in feats:
            # for every moment, the possible tag probabilities
            # feat is a 1D tensor with length tagset_size
            alphas_t = []
            # print 'feat: ', feat
            for next_tag in range(self.tagset_size):
                # print 'feat[next]: ', feat[next_tag]
                # feat[next_tag] is a 1D tensor, after view, it turn to 2D(1 * 1)
                # and then, it turn to 1 * tagset_size with the same value
                # tensor can not broadcast automatically
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # print 'emit_score: ', emit_score
                # transitions[next_tag] represents the probability from next_tag to other tags
                trans_score = self.transitions[next_tag].view(1, -1)
                # 1 * tagset_size
                next_tag_var = forward_var + trans_score + emit_score
                # get the alpha at moment t
                # max function can also
                alphas_t.append(log_sum_exp(next_tag_var))
            # alphas_t is a list of FloatTensor, not float
            # cat function transform 5 FloatTensor of size 1 to a FloatTensor of size 5
            forward_var = torch.cat(alphas_t).view(1, -1)
        # trans[STOP_TAG] means the probability from other tag to STOP
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        # print 'alpha: ', alpha
        # alpha: a FloatTensor of size 1
        return alpha

    def _viterbi_decode(self, feats):
        backpointers = []
        # analogous to forward
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = autograd.Variable(init_vvars)
        if self.use_gpu:
            forward_var = forward_var.cuda()
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                # this is the point different from forward precess
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            # backpointers is list of lists of Tensor of size 1
            backpointers.append(bptrs_t)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # best_tag_id is a number
        temp = terminal_var.data
        temp[0][self.tag_to_ix[STOP_TAG]] = -10000
        temp[0][self.tag_to_ix[START_TAG]] = -10000
        # print temp
        temp = torch.autograd.Variable(temp)
        best_tag_id = argmax(temp)
        path_score = temp[0][best_tag_id]
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, batch_sentence, sentence_length, batch_tags, batch_chars2, chars2_length):
        # batch_sentence, batch_tags: batch_size * max_length
        # transmit sentence to embedding layer and lstm layer, get features
        # features is a 3D tensor, batch * len(sentence) * self.tagset_size
        nll = 0
        batch_feats = self._get_lstm_features(batch_sentence, sentence_length, batch_chars2, chars2_length)
        assert batch_feats.size() == (self.batch_size, self.max_length, self.tagset_size)
        # print "feats: ", feats
        # CRF
        # the forward_score is the big Z
        # we want the bigger gold_score / forward_score is, the better.
        for feats, length, tags in zip(batch_feats, sentence_length, batch_tags):
            assert feats.size() == (self.max_length, self.tagset_size)
            # print(len(tags))
            # print(self.max_length)
            # print(length)
            forward_score = self._forward_alg(feats[:length])
        # calculate the score of the ground_truth, in CRF
            gold_score = self._score_sentence(feats[:length], tags[:length])

            nll += (forward_score - gold_score)
        # if not in log domain, there should be gold_score / forward_score
        # after taking the logarithm, add a minus
        return nll / len(sentence_length)

    def forward(self, batch_sentence, sentence_length, batch_chars2, chars2_length):
        # this function is defined to take a input and produce a output
        # when we call function  model(sentence), we call model.__call__(sentence),
        # and then, this forward is called
        # get the features of this sentence
        result = []
        batch_lstm_feats = self._get_lstm_features(batch_sentence, sentence_length, batch_chars2, chars2_length)
        # viterbi to get tag_seq
        for lstm_feats, l in zip(batch_lstm_feats, sentence_length):
            score, tag_seq = self._viterbi_decode(lstm_feats[:l])
            result.append((score, tag_seq))
        return result

