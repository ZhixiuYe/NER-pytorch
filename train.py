import os
import optparse
import itertools
import loader
import torch
import time
import re
import cPickle
import codecs
import sys
import visdom
import numpy as np
import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict

from torch.autograd import Variable
from utils import eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from model import BiLSTM_CRF
from loader import augment_with_pretrained
t = time.time()

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-T','--train', dest='train',
        default='dataset/eng.train', help="Train set location")
    parser.add_argument('-d','--dev', dest='dev',
        default='dataset/eng.testa', help="Dev set location")
    parser.add_argument('--score', dest='score',
        default='evaluation/temp/score.txt', help='score file location')
    parser.add_argument('-s','--tag_scheme', dest='tag_scheme',
        default='iobes', help='Tagging scheme (IOB or IOBES)')
    parser.add_argument('-l','--lower', dest='lower', type=int,
        default=0, help='Lowercase words (this will not affect character inputs)')
    parser.add_argument('-z','--zeros', dest='zeros', type=int,
        default=0, help='Replace digits with 0')
    parser.add_argument('-c','--char_dim', dest='char_dim', type=int,
        default=25, help='Character embedding dimension')
    parser.add_argument('-C','--char_lstm_dim', dest='char_lstm_dim', type=int,
        default=25, help='Character LSTM hidden layer size')
    parser.add_argument('-b','--char_bidirect', dest='char_bidirect', type=int,
        default=1, help='Use a bidirectional LSTM for characters')
    parser.add_argument('-w','--word_dim', dest='word_dim', type=int,
        default=100, help='Token embedding dimension')
    parser.add_argument('-W','--world_lstm_dim', dest='world_lstm_dim',
        default=100, help='Token LSTM hidden layer size')
    parser.add_argument('-B','--word_bidirect', dest='word_bidirect', type=int,
        default=1, help='Use a bidrectional LSTM for words')
    parser.add_argument('-p','--pre_emb', dest='pre_emb',
        default='models/glove.6B.100d.txt', help='Location of pretrained embeddings')
    parser.add_argument('-A','--all_emb', dest='all_emb', type=int,
        default=0, help='Load all embeddings')
    parser.add_argument('-a','--cap_dim', dest='cap_dim', type=int,
        default=0, help='Capitalization feature dimension(0 to disable')
    parser.add_argument('-f','--crf', dest='crf', type=int,
        default=1, help='Use CRF (0 to disable)')
    parser.add_argument('-D','--dropout', dest='dropout', type=float,
        default=0.5, help='Dropout on the input (0 = no dropout)')
    parser.add_argument('-r','--reload', dest='reload', type=int,
        default=0, help='Reload the last saved model')
    parser.add_argument('-g','--use_gpu', dest='use_gpu', type=int,
        default=0, help='Flag to use gpu')
    parser.add_argument('--loss', dest='loss', type=str,
        default='loss.txt', help='Loss file location')
    parser.add_argument('--name', dest='name', type=str,
        default='test', help='Model name')

    return parser.parse_args()


models_path = "models/"
opts = arguments()

parameters = OrderedDict()
parameters['tag_scheme'] = opts.tag_scheme
parameters['lower'] = opts.lower == 1
parameters['zeros'] = opts.zeros == 1
parameters['char_dim'] = opts.char_dim
parameters['char_lstm_dim'] = opts.char_lstm_dim
parameters['char_bidirect'] = opts.char_bidirect == 1
parameters['word_dim'] = opts.word_dim
parameters['word_lstm_dim'] = opts.word_lstm_dim
parameters['word_bidirect'] = opts.word_bidirect == 1
parameters['pre_emb'] = opts.pre_emb
parameters['all_emb'] = opts.all_emb == 1
parameters['cap_dim'] = opts.cap_dim
parameters['crf'] = opts.crf == 1
parameters['dropout'] = opts.dropout
parameters['reload'] = opts.reload == 1
parameters['name'] = opts.name

parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
use_gpu = parameters['use_gpu']

mapping_file = 'models/mapping.pkl'

name = parameters['name']
model_name = models_path + name #get_name(parameters)
tmp_model = model_name + '.tmp'


assert os.path.isfile(opts.train)
assert os.path.isfile(opts.dev)
assert os.path.isfile(opts.test)
assert parameters['char_dim'] > 0 or parameters['word_dim'] > 0
assert 0. <= parameters['dropout'] < 1.0
assert parameters['tag_scheme'] in ['iob', 'iobes']
assert not parameters['all_emb'] or parameters['pre_emb']
assert not parameters['pre_emb'] or parameters['word_dim'] > 0
assert not parameters['pre_emb'] or os.path.isfile(parameters['pre_emb'])

if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_temp):
    os.makedirs(eval_temp)
if not os.path.exists(models_path):
    os.makedirs(models_path)

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

train_sentences = loader.load_sentences(opts.train, lower, zeros)
dev_sentences = loader.load_sentences(opts.dev, lower, zeros)
test_sentences = loader.load_sentences(opts.test, lower, zeros)
test_train_sentences = loader.load_sentences(opts.test_train, lower, zeros)

update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)
update_tag_scheme(test_train_sentences, tag_scheme)

dico_words_train = word_mapping(train_sentences, lower)[0]
dico_words, word_to_id, id_to_word = augment_with_pretrained(
        dico_words_train.copy(),
        parameters['pre_emb'],
        list(itertools.chain.from_iterable(
            [[w[0] for w in s] for s in dev_sentences + test_sentences])
        ) if not parameters['all_emb'] else None
    )
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)

dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_train_data = prepare_dataset(
    test_train_sentences, word_to_id, char_to_id, tag_to_id, lower
)

print("%i / %i / %i sentences in train / dev / test." % (
    len(train_data), len(dev_data), len(test_data)))

all_word_embeds = {}
for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

print(len(all_word_embeds))
c_found = 0
c_lower = 0
c_zeros = 0
word_embeds = np.random.uniform(-1, 1, (len(word_to_id), opts.word_dim))

for w in word_to_id:
    if w in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w]
        c_found += 1
    elif w.lower() in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]
        c_lower += 1
    elif re.sub('\d', '0', w.lower()) in all_word_embeds:
        word_embeds[word_to_id[w]] = all_word_embeds[re.sub('\d', '0', w.lower())]
        c_zeros += 1
print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
print(('%i / %i (%.4f%%) words have been initialized with '
      'pretrained embeddings.') % (
    c_found + c_lower + c_zeros, len(word_to_id),
    100. * (c_found + c_lower + c_zeros) / len(word_to_id)
))
print(('%i found directly, %i after lowercasing, '
      '%i after lowercasing + zero.') % (
    c_found, c_lower, c_zeros
))

with open(mapping_file, 'wb') as f:
    mappings = {
        'word_to_id': word_to_id,
        'tag_to_id': tag_to_id,
        'char_to_id': char_to_id,
        'parameters': parameters,
        'word_embeds': word_embeds
    }
    cPickle.dump(mappings, f)

singletons = set([word_to_id[k] for k, v
                  in dico_words.items() if v == 1])

model = BiLSTM_CRF(len(word_to_id),
                   tag_to_id,
                   parameters['word_dim'],
                   parameters['word_lstm_dim'],
                   use_gpu=use_gpu,
                   char_to_ix=char_to_id,
                   pre_word_embeds=word_embeds,
                   use_crf=parameters['crf'])
                   # n_cap=4,
                   # cap_embedding_dim=10)
if parameters['reload']:
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()
learning_rate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
losses = []
loss = 0.0
best_dev_F = -1.0
best_test_F = -1.0
best_train_F = -1.0
all_F = [[0, 0, 0]]
plot_every = 1000
eval_every = 20000
count = 0
vis = visdom.Visdom()
sys.stdout.flush()


def evaluating(model, datas, best_F):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    for data in datas:
        ground_truth_id = data['tags']
        words = data['str_words']
        chars2 = data['chars']
        caps = data['caps']
        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj:
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c

        chars2_mask = Variable(torch.LongTensor(chars2_mask))
        dwords = Variable(torch.LongTensor(data['words']))
        dcaps = Variable(torch.LongTensor(caps))

        if use_gpu:
            val, out = model(dwords.cuda(), chars2_mask.cuda(), dcaps.cuda(),chars2_length, d)
        else:
            val, out = model(dwords, chars2_mask, dcaps, chars2_length, d)
        predicted_id = out
        for (word, true_id, pred_id) in zip(words, ground_truth_id, predicted_id):
            line = ' '.join([word, id_to_tag[true_id], id_to_tag[pred_id]])
            prediction.append(line)
            confusion_matrix[true_id, pred_id] += 1
        prediction.append('')
    predf = eval_temp + '/pred.' + name
    scoref = eval_temp + '/score.' + name

    with open(predf, 'wb') as f:
        f.write('\n'.join(prediction))

    os.system('%s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in codecs.open(scoref, 'r', 'utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)

    print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
        "ID", "NE", "Total",
        *([id_to_tag[i] for i in range(confusion_matrix.size(0))] + ["Percent"])
    ))
    for i in range(confusion_matrix.size(0)):
        print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
            str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
            *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
              ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
        ))
    return best_F, new_F, save

model.train(True)
for epoch in range(1, 10001):
    for i, index in enumerate(np.random.permutation(len(train_data))):
        tr = time.time()
        count += 1
        data = train_data[index]
        model.zero_grad()

        sentence_in = []
        for word in data['words']:
            if word in singletons and np.random.uniform() < 0.5:
                sentence_in.append(word_to_id['<UNK>'])
            else:
                sentence_in.append(word)
        sentence_in = Variable(torch.LongTensor(sentence_in))
        tags = data['tags']
        chars2 = data['chars']
        chars2_sorted = sorted(chars2, key=lambda p: len(p), reverse=True)
        # for batch char
        d = {}
        for i, ci in enumerate(chars2):
            for j, cj in enumerate(chars2_sorted):
                if ci == cj:
                    d[j] = i
                    continue
        chars2_length = [len(c) for c in chars2_sorted]
        char_maxl = max(chars2_length)
        chars2_mask = np.zeros((len(chars2_sorted), char_maxl), dtype='int')
        for i, c in enumerate(chars2_sorted):
            chars2_mask[i, :chars2_length[i]] = c

        targets = torch.LongTensor(tags)
        caps = Variable(torch.LongTensor(data['caps']))
        chars2_mask = Variable(torch.LongTensor(chars2_mask))
        if use_gpu:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in.cuda(),
                targets.cuda(), chars2_mask.cuda(), caps.cuda(), chars2_length, d)
        else:
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets,
                chars2_mask, caps, chars2_length, d)
        loss += neg_log_likelihood.data[0] / len(data['words'])
        neg_log_likelihood.backward()
        # print('model.word_embeds.weight.grad: ', model.word_embeds.weight.grad)
        # assert model.word_embeds.weight.grad == Variable(torch.zeros(28985, 100))
        # assert False
        torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if count % plot_every == 0:
            loss /= plot_every
            print(count, ': ', loss)
            if losses == []:
                losses.append(loss)
            losses.append(loss)
            text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
            losswin = 'loss_' + name
            textwin = 'loss_text_' + name
            vis.line(np.array(losses), X=np.array([plot_every*i for i in range(len(losses))]),
                 win=losswin, opts={'title': losswin, 'legend': ['loss']})
            vis.text(text, win=textwin, opts={'title': textwin})
            loss = 0.0

        if count % (eval_every) == 0 and count > (eval_every * 20) or \
                count % (eval_every*4) == 0 and count < (eval_every * 20):
            model.train(False)
            best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
            best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F)
            if save:
                torch.save(model.state_dict(), model_name)
            best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)
            sys.stdout.flush()

            all_F.append([new_train_F, new_dev_F, new_test_F])
            Fwin = 'F-score of {train, dev, test}_' + name
            vis.line(np.array(all_F), win=Fwin,
                 X=np.array([eval_every*i for i in range(len(all_F))]),
                 opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
            model.train(True)

print(time.time() - t)

plt.plot(losses)
plt.show()