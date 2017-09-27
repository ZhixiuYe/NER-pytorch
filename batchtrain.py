# coding=utf-8
from __future__ import print_function
import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader
import torch
import time
import cPickle
import copy
from torch.autograd import Variable
import matplotlib.pyplot as plt
import codecs
import sys
import random
import re
import visdom


from utils import eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import random_batch
from loader import augment_with_pretrained
from loader import get_batch


from batchmodel import batch_BiLSTM_CRF

t = time.time()
name = 'batch8_Adam'
models_path = "models/"
model_name = models_path + name #get_name(parameters)
tmp_model = model_name + '.tmp'

optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="dataset/eng.train",
    help="Train set location"
)
optparser.add_option(
    "-d", "--dev", default="dataset/eng.testb",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="dataset/eng.testa",
    help="Test set location"
)
optparser.add_option(
    '--test_train', default='dataset/eng.train54019',
    help='test train'
)
optparser.add_option(
    '--score', default='evaluation/temp/batch_score.txt',
    help='score file location'
)
optparser.add_option(
    "-s", "--tag_scheme", default="iobes",
    help="Tagging scheme (IOB or IOBES)"
)
optparser.add_option(
    "-l", "--lower", default="0",
    type='int', help="Lowercase words (this will not affect character inputs)"
)
optparser.add_option(
    "-z", "--zeros", default="0",
    type='int', help="Replace digits with 0"
)
optparser.add_option(
    "-c", "--char_dim", default="25",
    type='int', help="Char embedding dimension"
)
optparser.add_option(
    "-C", "--char_lstm_dim", default="25",
    type='int', help="Char LSTM hidden layer size"
)
optparser.add_option(
    "-b", "--char_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for chars"
)
optparser.add_option(
    "-w", "--word_dim", default="100",
    type='int', help="Token embedding dimension"
)
optparser.add_option(
    "-W", "--word_lstm_dim", default="100",
    type='int', help="Token LSTM hidden layer size"
)
optparser.add_option(
    "-B", "--word_bidirect", default="1",
    type='int', help="Use a bidirectional LSTM for words"
)
optparser.add_option(
    "-p", "--pre_emb", default="models/glove.6B.100d.txt",
    help="Location of pretrained embeddings"
)
optparser.add_option(
    "-A", "--all_emb", default="0",
    type='int', help="Load all embeddings"
)
optparser.add_option(
    "-a", "--cap_dim", default="0",
    type='int', help="Capitalization feature dimension (0 to disable)"
)
optparser.add_option(
    "-f", "--crf", default="0",
    type='int', help="Use CRF (0 to disable)"
)
optparser.add_option(
    "-D", "--dropout", default="0.5",
    type='float', help="Droupout on the input (0 = no dropout)"
)
# optparser.add_option(
#     "-L", "--lr_method", default="sgd-lr_.005",
#     help="Learning method (SGD, Adadelta, Adam..)"
# )
optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)
optparser.add_option(
    "-g", '--use_gpu', default='1',
    type='int', help='whether or not to ues gpu'
)
optparser.add_option(
    '--batch_size', default='8',
    type='int', help='batch size'
)

optparser.add_option(
    '--loss', default='loss.txt',
    help='loss file location'
)

opts = optparser.parse_args()[0]

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
# parameters['lr_method'] = opts.lr_method
parameters['use_gpu'] = opts.use_gpu == 1 and torch.cuda.is_available()
parameters['batch_size'] = opts.batch_size
use_gpu = parameters['use_gpu']
mapping_file = 'models/batch_mapping.pkl'



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


words_num = len(word_to_id)
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower)

dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_train_data = prepare_dataset(
    test_train_sentences, word_to_id, char_to_id, tag_to_id, lower
)


all_word_embeds = {}
for i, line in enumerate(codecs.open(opts.pre_emb, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == parameters['word_dim'] + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])


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

batch_size = parameters['batch_size']


learning_rate = 0.005
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-4)
losses = []
los = 0.0
loss = 0.0
old_dev_F = -1.0
old_test_F = -1.0
plot_every = 100
eval_every = 2000
vis = visdom.Visdom()
best_train_F = -1
best_dev_F = -1
best_test_F = -1
all_F = all_F = [[0, 0, 0]]
sys.stdout.flush()




model = batch_BiLSTM_CRF(len(word_to_id),
                             tag_to_id,
                             parameters['word_dim'],
                             parameters['word_lstm_dim'],
                             use_gpu=use_gpu,
                             char_to_ix=char_to_id,
                             pre_word_embeds=word_embeds)
    # model = torch.nn.DataParallel(model, device_ids=[1,3], output_device=[1])
if parameters['reload']:
    model.load_state_dict(torch.load(model_name))
if use_gpu:
    model.cuda()
optimizer = torch.optim.Adam(model.parameters())


def evaluating(model, datas, best_F):
    prediction = []
    save = False
    new_F = 0.0
    confusion_matrix = torch.zeros((len(tag_to_id) - 2, len(tag_to_id) - 2))
    start = -batch_size
    while start < len(datas):
        start += batch_size
        input_padded, input_lengths, target_padded, target_lengths, chars2_seqs_padded,\
        chars2_seqs_lengths = get_batch(start, batch_size, datas, singletons)

        if input_padded == []:
            break

        # ground_truth_id = data['tags']
        # words = data['str_words']
        # chars2 = data['chars']
        # caps = data['caps']
        # input_padded = torch.autograd.Variable(torch.LongTensor(data['words']).view(1, -1))
        # input_lengths = [input_padded.shape[1]]
        if use_gpu:
            V_input_padded = Variable(torch.LongTensor(input_padded)).cuda()
            # target_padded = torch.LongTensor(target_padded).cuda()
        else:
            V_input_padded = Variable(torch.LongTensor(input_padded))
            # target_padded = torch.LongTensor(target_padded)

        result = model(V_input_padded, input_lengths, chars2_seqs_padded, chars2_seqs_lengths)

        # predicted_id = target_padded
        for (score, pred_ids), true_ids, l, sentence in zip(result, target_padded, target_lengths, input_padded):
                # print(score, tags)
                # print('   ', target[:l])
            # assert len(true_ids) == l

            for (word, true_id, pred_id) in zip(sentence[:l], true_ids[:l], pred_ids[:l]):
                line = ' '.join([id_to_word[word], id_to_tag[true_id], id_to_tag[pred_id]])
                prediction.append(line)
                confusion_matrix[true_id, pred_id] += 1
            prediction.append('')
    predf = eval_temp + '/pred.txt'
    scoref = eval_temp + '/score.txt'

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

    # print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
    #     "ID", "NE", "Total",
    #     *([id_to_tag[i] for i in xrange(confusion_matrix.size(0))] + ["Percent"])
    # ))
    # for i in xrange(confusion_matrix.size(0)):
    #     print(("{: >2}{: >7}{: >7}%s{: >9}" % ("{: >7}" * confusion_matrix.size(0))).format(
    #         str(i), id_to_tag[i], str(confusion_matrix[i].sum()),
    #         *([confusion_matrix[i][j] for j in range(confusion_matrix.size(0))] +
    #           ["%.3f" % (confusion_matrix[i][i] * 100. / max(1, confusion_matrix[i].sum()))])
    #     ))
    return best_F, new_F, save


    # if epoch == 100:
    #     learning_rate /= 2
    #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=1e-4)

for i in range(1, 100000):
        input_padded, input_lengths, target_padded, target_lengths, chars2_seqs_padded, \
        chars2_seqs_lengths = random_batch(batch_size, train_data, singletons=singletons)

        if use_gpu:
            input_padded = Variable(torch.LongTensor(input_padded)).cuda()
            target_padded = torch.LongTensor(target_padded).cuda()
        else:
            input_padded = Variable(torch.LongTensor(input_padded))
            target_padded = torch.LongTensor(target_padded)

        model.zero_grad()
        neg_log_likelihood = model.neg_log_likelihood(input_padded, input_lengths,
                                                  target_padded,
                                                  chars2_seqs_padded,
                                                  chars2_seqs_lengths)
        los = neg_log_likelihood.data[0]
        loss += los
        neg_log_likelihood.backward()
        optimizer.step()
        if i % plot_every == 0:
            loss /= plot_every
            print(i*batch_size, ': ', loss)
            if losses == []:
                losses.append(loss)
            losses.append(loss)
            text = '<p>' + '</p><p>'.join([str(l) for l in losses[-9:]]) + '</p>'
            losswin = 'loss_batch(line)'
            textwin = 'loss2_batch(number)'
            vis.line(np.array(losses), X=np.array([plot_every*j for j in range(len(losses))]),
                 win=losswin, opts={'title': losswin, 'legend': ['loss']})
            vis.text(text, win=textwin, opts={'title': textwin})
            loss = 0.0

        if i % (eval_every / 2) == 0 and i >= (eval_every * 50) or \
           i % eval_every == 0 and i < (eval_every * 50):
                model.train(False)
                # torch.save(model.state_dict(), tmp_model)
                best_train_F, new_train_F, _ = evaluating(model, test_train_data, best_train_F)
                best_dev_F, new_dev_F, save = evaluating(model, dev_data, best_dev_F)
                if save:
                    torch.save(model.state_dict(), model_name)
                best_test_F, new_test_F, _ = evaluating(model, test_data, best_test_F)
                sys.stdout.flush()

                all_F.append([new_train_F, new_dev_F, new_test_F])
                Fwin = 'F-score of {train, dev, test}_' + name
                vis.line(np.array(all_F), win=Fwin,
                         X=np.array([eval_every * i for i in range(len(all_F))]),
                         opts={'title': Fwin, 'legend': ['train', 'dev', 'test']})
                model.train(True)



    # if epoch % 50 == 0:
    #     result = model(input_padded, input_lengths, chars2_seqs_padded, chars2_seqs_lengths)
    #
    #     for (score, tags), target, l in zip(result, target_padded, target_lengths):
    #         print(score, tags)
    #         print('   ', target[:l])

    # assert False

    # if epoch % 5 == 0:
    #     input_padded, input_lengths, target_padded, target_lengths, \
    #     chars2_seqs_padded, chars2_seqs_lengths = random_batch(1, train_data)
    #     print(input_padded, input_lengths, target_padded, target_lengths,
    #     chars2_seqs_padded, chars2_seqs_lengths)
    #     if use_gpu:
    #         input_padded = Variable(torch.LongTensor(input_padded)).cuda()
    #         target_padded = torch.LongTensor(target_padded).cuda()
    #     else:
    #         input_padded = Variable(torch.LongTensor(input_padded))
    #         target_padded = torch.LongTensor(target_padded)
    #     result = model(input_padded, input_lengths, chars2_seqs_padded, chars2_seqs_lengths)
    #     for (score, tags), target, l in zip(result, target_padded, target_lengths):
    #         print(score, tags)
    #         print('   ', target[:l])
    # assert False



    #     os.system('python eval.py -i %s > %s' % (opts.dev, opts.score))
    #     eval_lines = [l.rstrip() for l in codecs.open(opts.score, 'r', 'utf8')]
    #
    #     for i, line in enumerate(eval_lines):
    #         print(line)
    #         if i == 1:
    #             new_dev_F = float(line.strip().split()[-1])
    #             if new_dev_F > old_dev_F:
    #                 print('the best dev F is ', new_dev_F)
    #                 torch.save(model.state_dict(), model_name)
    #                 old_dev_F = new_dev_F
    #     # sys.stdout.flush()
    #
    #     os.system('python eval.py -i %s > %s' % (opts.test, opts.score))
    #     eval_lines = [l.rstrip() for l in codecs.open(opts.score, 'r', 'utf8')]
    #     for i, line in enumerate(eval_lines):
    #         print(line)
    #         if i == 1:
    #             new_test_F = float(line.strip().split()[-1])
    #             if new_test_F > old_test_F:
    #                 print('the best test F is ', new_test_F)
    #                 # torch.save(model.state_dict(), models_path + '/models1.pkl')
    #                 old_test_F = new_test_F
    #     # sys.stdout.flush()
    #
    #     with open(opts.loss, 'wb') as f:
    #         cPickle.dump(losses, f)
    # # sys.stdout.flush()

print(time.time() - t)

# torch.save(model.state_dict(), models_path + '/models1.pkl')

plt.plot(losses)
plt.show()
