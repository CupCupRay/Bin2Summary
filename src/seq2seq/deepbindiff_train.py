import os
import ast
import sys
import json
import time
import string
import random
import gensim
import pickle
import platform
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt  # matplotlib.use('agg')
import tensorflow.keras.backend as K
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# import sklearn.metrics

## conda env export > environment.yaml
## IMPORTANT!!! Python >= 3.6 !!!
## IMPORTANT!!! Tensorflow-gpu >= 2.1.0 !!!
## IMPORTANT!!! Gensim >= 4.0.1 !!!

# CUDA_VISIBLE_DEVICES=0,1,2,3
# tf.config.threading.set_intra_op_parallelism_threads(8)
# tf.config.threading.set_inter_op_parallelism_threads(8)
print('Using the Tensorflow version', tf.version)
print('GPU:', tf.config.list_physical_devices('GPU'))
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)

####################### CONFIGURATIONS #######################
### Data Related
BB_MAX_NUM = 300  # CONST
BB_VECTOR_DIM = 128  # CONST == 128
COMMENT_MAX_LEN = 15
WORD_EMBEDDING_DIM = 128  # CONST == 128
BATCH_SIZE_PER_REPLICA = 256
# BATCH_SIZE_PER_REPLICA = 16
TRAIN_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
STEP_PER_EPOCH = 256
# STEP_PER_EPOCH = 4
# MAX_TRAIN_SIZE = TRAIN_BATCH_SIZE * STEP_PER_EPOCH

### Network & Training Related
# REPEAT = 1
EPOCHS = 1000
HIDDEN_DIM = 512
NETWORK_DEPTH = 6  # Need >= 2
DROPOUT = 0.2
LR = 0.0002
MIN_LR = 0.00001
CALL_BACK_MONITOR = 'val_loss'
####################### CONFIGURATIONS #######################
file_count = 0

####################### Service Setting #######################
normal_path = './'

####################### Service Setting #######################


def PrintConfig():
    config_info = ('input_length=BB_MAX_NUM=' + str(BB_MAX_NUM) + ', ' +
                   'input_dim=BB_VECTOR_DIM=' + str(BB_VECTOR_DIM) + ', ' +
                   'output_length=COMMENT_MAX_LEN=' + str(COMMENT_MAX_LEN) + ', ' +
                   'output_dim=WORD_EMBEDDING_DIM=' + str(WORD_EMBEDDING_DIM) + ', ' +
                   'training_batch_size=TRAIN_BATCH_SIZE=' + str(TRAIN_BATCH_SIZE) + ', ' +
                   'STEP_PER_EPOCH=' + str(STEP_PER_EPOCH) + ', ' +
                   # 'MAX_TRAIN_SIZE=' + str(MAX_TRAIN_SIZE) + ', ' +

                   # 'REPEAT=' + str(REPEAT) + ', ' +
                   'EPOCHS=' + str(EPOCHS) + ', ' +
                   'NETWORK_DEPTH=' + str(NETWORK_DEPTH) + ', ' +
                   'hidden_dim=HIDDEN_DIM=' + str(HIDDEN_DIM) + ', ' +
                   'DROPOUT=' + str(DROPOUT) +
                   '\n--------------------------------------------------------------------------\n')
    return config_info


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def prepare_data(path, only_test=False, my_seed=42):
    train_com_batch, train_emb_batch = dict(), dict()
    val_com_batch, val_emb_batch = dict(), dict()
    test_com_batch, test_emb_batch = dict(), dict()  # Batches of com/emb, dict [pack: list [com_1, com_2], ...]
    whole_com_batch = []  # Batches of comments, list [comment_1 <list<str>>, comment_2 <list<str>>, ...]

    print("Begin to extract the features (been preprocessed).")
    for root, dirs, files in os.walk(path):
        if not root.endswith('/'): root = root + '/'
        if root == path or root == path + '/':
            for file in files:
                if file.endswith('embeddings.record') or file.endswith('comment.record'):
                    os.remove(path + file)

    pack_file_num = dict()
    for root, dirs, files in os.walk(path):
        if not root.endswith('/'): root = root + '/'
        if root == path or root == path + '/': continue
        for file in files:
            if file.endswith('embeddings.record'):
                pack_name = file[:file.find('_')]
                if pack_name not in pack_file_num:
                    pack_file_num[pack_name] = 1
                else:
                    pack_file_num[pack_name] = pack_file_num[pack_name] + 1

    current_pack_file = dict()
    for root, dirs, files in os.walk(path):
        if not root.endswith('/'): root = root + '/'
        if root == path or root == path + '/': continue

        com_postfix = '_comment.record'
        random.shuffle(files)
        for file in files:
            if file.endswith('embeddings.record'):
                pack_name = file[:file.find('_')]
                emb_postfix = file[file.rfind('_'):]
                if pack_name not in current_pack_file:
                    current_pack_file[pack_name] = 1
                else:
                    current_pack_file[pack_name] = current_pack_file[pack_name] + 1

                if current_pack_file[pack_name] <= int(pack_file_num[pack_name] * 0.1):
                    with open(path + 'test' + emb_postfix, mode='a') as record_file:
                        for line in open(root + file, mode='r'):
                            record_file.write(line)
                    with open(path + 'test' + com_postfix, mode='a') as record_file:
                        for line in open(root + file.replace(emb_postfix, com_postfix), mode='r'):
                            record_file.write(line)
                elif current_pack_file[pack_name] <= int(pack_file_num[pack_name] * 0.2):
                    with open(path + 'val' + emb_postfix, mode='a') as record_file:
                        for line in open(root + file, mode='r'):
                            record_file.write(line)
                    with open(path + 'val' + com_postfix, mode='a') as record_file:
                        for line in open(root + file.replace(emb_postfix, com_postfix), mode='r'):
                            record_file.write(line)
                else:
                    with open(path + 'train' + emb_postfix, mode='a') as record_file:
                        for line in open(root + file, mode='r'):
                            record_file.write(line)
                    with open(path + 'train' + com_postfix, mode='a') as record_file:
                        for line in open(root + file.replace(emb_postfix, com_postfix), mode='r'):
                            record_file.write(line)

    E_Exists, C_Exists = False, False
    for root, dirs, files in os.walk(path):
        if not root.endswith('/'): root = root + '/'
        if root == path or root == path + '/':
            for file in files:
                if file.endswith('comment.record'):
                    C_Exists = True
                if file.endswith('embeddings.record'):
                    E_Exists = True
    if not E_Exists or not C_Exists:
        for _ in range(10):
            print('[ERROR! CANNOT FIND EMBEDDING AND COMMENT!]')
        return -1

    whole_com_file = open(path + 'comment.ref', mode='r')  # Collect the Word2vec training set
    for whole_com in whole_com_file:
        whole_com = whole_com.replace('\n', '')
        elements = whole_com.split(' => ', 1)
        if len(elements) < 2: continue
        whole_com_data = elements[1]
        whole_com_data = json.loads(whole_com_data)
        if not whole_com_data: continue
        if type(whole_com_data) == str:
            whole_com_data = whole_com_data.lower()
            temp_com = ''
            for c in whole_com_data:
                if 'a' <= c <= 'z' or c == ' ':
                    temp_com = temp_com + c
                else:
                    continue
            whole_com_data = temp_com.split(' ')
        assert type(whole_com_data) == list
        whole_com_batch.append(whole_com_data)
    whole_com_file.close()

    packs = []
    for root, dirs, files in os.walk(path):
        if not root.endswith('/'): root = root + '/'
        if root != path and root != path + '/': continue
        for file in files:
            if file.endswith('embeddings.record') or file.endswith('comment.record'):
                if file.find('_') == -1:
                    name = ''
                else:
                    name = file[:file.find('_')] + '_'
                if name not in packs:
                    packs.append(name)

    for p in packs:
        print('Preparing the embedding data in', path + p)

        com_file = open(path + p + 'comment.record', mode='r')
        emb_file = open(path + p + 'embeddings.record', mode='r')
        temp_com_batch, temp_emb_batch = [], []

        for com, emb in zip(com_file, emb_file):
            com = com.replace('\n', '')
            emb = emb.replace('\n', '')
            com_type, com_data = com.split(' => ')
            if not (com_data.startswith('[') and com_data.endswith(']')):
                com_data = com_data.replace('.', '').replace(',', '')
                com_data = com_data.lower().split(' ')
            else:
                com_data = json.loads(com_data)
            if not com_data: continue
            assert type(com_data) == list
            for ele in com_data:
                assert type(ele) == str
            while '' in com_data: com_data.remove('')

            emb_data = json.loads(emb)
            assert type(emb_data) == list
            for ele in emb_data:
                assert type(ele) == list
                for num in ele:
                    assert type(num) == float

            if len(com_data) > COMMENT_MAX_LEN:
                com_data = com_data[:COMMENT_MAX_LEN]

            emb_data = preprocess_embeddings(emb_data, com_data, padding=True)
            if emb_data.any():
                temp_com_batch.append(com_data)
                temp_emb_batch.append(emb_data)
        emb_file.close()
        com_file.close()
        assert len(temp_com_batch) == len(temp_emb_batch)

        # Shuffle
        shuffle_com_batch = []
        shuffle_emb_batch = []
        index = [i for i in range(len(temp_com_batch))]
        # random.Random(my_seed).shuffle(index)
        for i in index:
            shuffle_com_batch.append(temp_com_batch[i])
            shuffle_emb_batch.append(temp_emb_batch[i])
        assert len(shuffle_com_batch) == len(shuffle_emb_batch)

        if 'train' in p:
            train_com_batch[p] = shuffle_com_batch
            train_emb_batch[p] = shuffle_emb_batch
        elif 'val' in p:
            val_com_batch[p] = shuffle_com_batch
            val_emb_batch[p] = shuffle_emb_batch
        elif 'test' in p:
            test_com_batch[p] = shuffle_com_batch
            test_emb_batch[p] = shuffle_emb_batch

    assert len(train_com_batch) == len(train_emb_batch)
    assert len(val_com_batch) == len(val_emb_batch)
    assert len(test_com_batch) == len(test_emb_batch)

    out_train_com, out_train_emb = [], []
    for ele in train_com_batch:
        out_train_com = out_train_com + train_com_batch[ele]
    for ele in train_emb_batch:
        out_train_emb = out_train_emb + train_emb_batch[ele]

    out_val_com, out_val_emb = [], []
    for ele in val_com_batch:
        out_val_com = out_val_com + val_com_batch[ele]
    for ele in val_emb_batch:
        out_val_emb = out_val_emb + val_emb_batch[ele]

    out_test_com, out_test_emb = [], []
    for ele in test_com_batch:
        out_test_com = out_test_com + test_com_batch[ele]
    for ele in test_emb_batch:
        out_test_emb = out_test_emb + test_emb_batch[ele]

    if not only_test:
        return out_train_com, out_train_emb, out_val_com, out_val_emb, out_test_com, out_test_emb, whole_com_batch
    elif only_test:
        return train_com_batch, test_com_batch, test_emb_batch


def comment_padding(com_data):
    new_com = com_data.copy()
    assert len(new_com) <= COMMENT_MAX_LEN
    if len(new_com) == COMMENT_MAX_LEN:
        new_com[-1] = '<EOS>'
    else:
        new_com.append('<EOS>')
    while len(new_com) < COMMENT_MAX_LEN: new_com.append('<PAD>')
    new_com = ['<GO>'] + new_com
    assert len(new_com) == COMMENT_MAX_LEN + 1
    return new_com


def preprocess_embeddings(embeddings, comment, padding=True):
    global file_count
    file_count = file_count + 1
    while '' in embeddings: embeddings.remove('')
    new_embeddings = np.zeros((1, BB_VECTOR_DIM))

    for emb in embeddings:
        for token in emb:
            if str(token) == 'nan' or str(token) == 'Nan' or str(token) == 'NaN' or \
                    str(token) == 'Inf' or str(token) == 'inf':
                return new_embeddings

    count = len(embeddings)
    if padding and BB_MAX_NUM is not None:
        if count > BB_MAX_NUM:
            new_embeddings = np.array(embeddings[:BB_MAX_NUM])
        else:
            if len(embeddings) < BB_MAX_NUM:
                new_embeddings = np.concatenate((np.array(embeddings),
                                                 np.zeros((BB_MAX_NUM - len(embeddings), BB_VECTOR_DIM))))
            else:
                new_embeddings = np.array(embeddings)

    else:
        new_embeddings = np.array(embeddings)

    return new_embeddings


def decode_vec2word(vs, w2v_model, mode='Normal'):
    sentence = []
    for temp_vec in vs:
        temp_vec = np.array(temp_vec)
        prediction, prob = w2v_model.most_similar(positive=[temp_vec], topn=1)[0]
        sentence.append(prediction)
    cleaned_sentence = []
    for word in sentence:
        if word == '<EOS>' and mode == 'Normal': break
        if word == '<PAD>' and mode == 'Normal': continue
        if word == '<GO>' and mode == 'Normal': continue
        cleaned_sentence.append(word)
    return ' '.join(cleaned_sentence)


def generate_reference(comment_sentence, syn):
    reference_batch = [comment_sentence]
    for s in syn:
        temp_words = []
        Flag = False
        for word in comment_sentence:
            if s == word:
                temp_words.append(syn[s])
                Flag = True
            else:
                temp_words.append(word)
        if Flag: reference_batch.append(temp_words)
    return reference_batch


def customize_model():
    with strategy.scope():
        encoder_input = tf.keras.layers.Input(shape=(BB_MAX_NUM, BB_VECTOR_DIM), name='Encoder_In')
        decoder_input = tf.keras.layers.Input(shape=(COMMENT_MAX_LEN, WORD_EMBEDDING_DIM), name='Decoder_In')
        encoder_gru = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, return_state=True,
                                          name="En_Bi_0", dropout=DROPOUT)  # , dropout=DROPOUT
        x, state = encoder_gru(encoder_input)

        for num in range(NETWORK_DEPTH - 2):
            encoder_gru = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, return_state=False,
                                              name="En_Bi_" + str(num + 1), dropout=DROPOUT)  # , dropout=DROPOUT
            x = encoder_gru(x)
        encoder_gru = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, return_state=True,
                                          name="En_Bi_" + str(NETWORK_DEPTH - 1), dropout=DROPOUT)
        x, state = encoder_gru(x)
        encoder_states = [state]

        decoder_gru = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, return_state=True,
                                          name="De_0", dropout=DROPOUT)  # , dropout=DROPOUT
        y, _ = decoder_gru(decoder_input, initial_state=encoder_states)

        for num in range(NETWORK_DEPTH - 2):
            decoder_gru = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, return_state=False,
                                              name="De_" + str(num + 1), dropout=DROPOUT)
            y = decoder_gru(y)
        decoder_gru = tf.keras.layers.GRU(HIDDEN_DIM, return_sequences=True, return_state=False,
                                          name="De_" + str(NETWORK_DEPTH - 1), dropout=DROPOUT)
        y = decoder_gru(y)

        decoder_output = tf.keras.layers.AdditiveAttention(name="Attention")([y, x])
        decoder_output = tf.keras.layers.Concatenate()([y, decoder_output])
        decoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense
                                                         (units=WORD_EMBEDDING_DIM), name="Out")(decoder_output)

        model = tf.keras.models.Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

        opt = tf.keras.optimizers.Adam(learning_rate=LR)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mean_absolute_error'])

    return model


def generate_data(x, y=None, z=None, batch_size=1):
    if y is None and z is None:
        for data in x:
            yield data
    elif z is None:
        for en, de in zip(x, y):
            yield [en, de]
    elif batch_size != 0:
        assert len(x) == len(y) == len(z)
        count = 0
        while count + batch_size < len(x):
            yield {'Encoder_In': x[count: count + batch_size],
                   'Decoder_In': y[count: count + batch_size]}, \
                  z[count: count + batch_size]
            count += batch_size


def combine_en_de_input(en, de):
    return en, de


def data_augment(in_list):
    temp_list = in_list
    for i in range(1, 20):
        temp_list = temp_list + in_list
    return temp_list


def main(Word2vec_train_set='Load', model_load=None, On_Service=False):
    path_prefix = normal_path
    timeRecord = str(time.asctime(time.localtime(time.time()))).replace(' ', '_').replace(':', '_')
    result_f = open(path_prefix + 'results/' + timeRecord + '.log', mode='w')

    result_f.write('With the data path ' + file_path + '\n')

    train_comment_batch, train_embedding_batch, \
    val_comment_batch, val_embedding_batch, \
    test_comment_batch, test_embedding_batch, whole_comment_batch = prepare_data(path_prefix + file_path)

    if not train_embedding_batch or not val_embedding_batch or not test_embedding_batch:
        print("Please prepare the comment.record file and embeddings.record files.")
        return -1

    padded_train_comment = []
    for temp_com in train_comment_batch:
        padded_train_comment.append(comment_padding(temp_com))

    padded_val_comment = []
    for temp_com in val_comment_batch:
        padded_val_comment.append(comment_padding(temp_com))

    padded_test_comment = []
    for temp_com in test_comment_batch:
        padded_test_comment.append(comment_padding(temp_com))

    assert len(train_comment_batch) == len(train_embedding_batch)
    assert len(val_comment_batch) == len(val_embedding_batch)
    assert len(test_comment_batch) == len(test_embedding_batch)
    print("And %d <comment, snippet> pairs for training." % (len(train_embedding_batch)))
    print("And %d <comment, snippet> pairs for validation." % (len(val_embedding_batch)))
    print("And %d <comment, snippet> pairs for testing." % (len(test_embedding_batch)))
    result_f.write('And ' + str(len(train_embedding_batch)) + ' <comment, snippet> pairs for training.\n')
    result_f.write('And ' + str(len(val_embedding_batch)) + ' <comment, snippet> pairs for validation.\n')
    result_f.write('And ' + str(len(test_embedding_batch)) + ' <comment, snippet> pairs for testing.\n')

    my_Gensim_word2vec_model = None
    if 'New' not in Word2vec_train_set and model_load is not None and \
            os.path.exists(path_prefix + 'results/' + model_load + '_word2vec.model'):
        my_Gensim_word2vec_model = gensim.models.Word2Vec.load(
            path_prefix + 'results/' + model_load + '_word2vec.model')
        print('Successfully load the gensim.word2vec for comments.')
        result_f.write('Successfully load the ' + model_load + ' gensim.word2vec for comments.\n')
    elif Word2vec_train_set == 'Load' and os.path.exists(path_prefix + 'results/My_comment_word2vec.model'):
        my_Gensim_word2vec_model = gensim.models.Word2Vec.load(path_prefix + 'results/My_comment_word2vec.model')
        print('Successfully load the gensim.word2vec for comments.')
        result_f.write('Successfully load the gensim.word2vec for comments.\n')
    elif 'New' in Word2vec_train_set:
        temp_word2vec_train = whole_comment_batch + padded_train_comment + padded_test_comment
        print('There are ' + str(len(temp_word2vec_train)) + ' complete comments in Word2vec training.')
        my_Gensim_word2vec_model = gensim.models.Word2Vec(temp_word2vec_train, vector_size=WORD_EMBEDDING_DIM, hs=1,
                                                          min_count=1, cbow_mean=0, window=5, epochs=5000, workers=16)
        my_Gensim_word2vec_model.save(path_prefix + 'results/' + timeRecord + '_word2vec.model')
        print('Complete the training of gensim.word2vec for comments.')

    if my_Gensim_word2vec_model is None:
        print('Error with Word2vec model!!!')
        return -1
    print('The vocab capacity is', len(my_Gensim_word2vec_model.wv.index_to_key))
    result_f.write('The vocab capacity is ' + str(len(my_Gensim_word2vec_model.wv.index_to_key)) + '\n')
    result_f.close()

    train_comments_vec = []  # The comments in form of index, list [comment_1 'word_vec', comment_2 'word_vec', ...]
    for comment in train_comment_batch:
        words_vec = []
        new_comment = comment_padding(comment)
        assert new_comment[0] == '<GO>'
        for word in new_comment[1:]:
            words_vec.append(my_Gensim_word2vec_model.wv.word_vec(word))
        assert len(words_vec) == COMMENT_MAX_LEN
        train_comments_vec.append(words_vec)
    print('Total ' + str(len(train_comments_vec)) + ' Comments in training set into vectors')

    val_comments_vec = []  # The comments in form of index, list [comment_1 'word_vec', comment_2 'word_vec', ...]
    for comment in val_comment_batch:
        words_vec = []
        new_comment = comment_padding(comment)
        assert new_comment[0] == '<GO>'
        for word in new_comment[1:]:
            words_vec.append(my_Gensim_word2vec_model.wv.word_vec(word))
        assert len(words_vec) == COMMENT_MAX_LEN
        val_comments_vec.append(words_vec)
    print('Total ' + str(len(val_comments_vec)) + ' Comments in validation set into vectors')

    test_individual_input = []
    for embeddings in test_embedding_batch:
        test_individual_input.append(np.array([np.array(emb) for emb in embeddings]))
    test_input = np.array([ind for ind in test_individual_input])

    # return -1

    my_model = None
    plt.figure()

    train_output = np.array(train_comments_vec)
    val_output = np.array(val_comments_vec)
    print("Final Output data shape: ", train_output.shape)

    assert len(train_embedding_batch) == len(train_output)
    assert len(val_embedding_batch) == len(val_output)
    train_decoder_input = np.insert(train_output, 0, my_Gensim_word2vec_model.wv.word_vec('<GO>'), axis=1)[:, :, :]
    train_decoder_input = train_decoder_input[:, :-1, :]
    val_decoder_input = np.insert(val_output, 0, my_Gensim_word2vec_model.wv.word_vec('<GO>'), axis=1)[:, :, :]
    val_decoder_input = val_decoder_input[:, :-1, :]

    with open(path_prefix + 'results/' + timeRecord + '.log', mode='a') as result_f:
        for vec_1, vec_2 in random.sample(list(zip(train_output, train_decoder_input)), 10):
            result_f.write('Example Decoder output in Seq: ' +
                           decode_vec2word(vec_1.tolist(), my_Gensim_word2vec_model.wv, mode='Full') + '\n')
            result_f.write('Example Decoder input in Seq: ' +
                           decode_vec2word(vec_2.tolist(), my_Gensim_word2vec_model.wv, mode='Full') + '\n')
    assert train_output.shape == train_decoder_input.shape
    assert val_output.shape == val_decoder_input.shape

    seed = random.randrange(sys.maxsize)

    signature = ({'Encoder_In': tf.TensorSpec(shape=(BB_MAX_NUM, BB_VECTOR_DIM), dtype=tf.float64),
                  'Decoder_In': tf.TensorSpec(shape=(COMMENT_MAX_LEN, WORD_EMBEDDING_DIM), dtype=tf.float64)},
                 tf.TensorSpec(shape=(COMMENT_MAX_LEN, WORD_EMBEDDING_DIM), dtype=tf.float64))

    class my_generator:
        def __init__(self, en_in, de_in, out):
            self.Encoder_In = en_in
            self.Decoder_In = de_in
            self.Out = out
            assert len(self.Encoder_In) == len(self.Decoder_In) == len(self.Out)

        def __call__(self, *args, **kwargs):
            for x, y, z in zip(self.Encoder_In, self.Decoder_In, self.Out):
                yield {'Encoder_In': x, 'Decoder_In': y}, z

    train_dataset = tf.data.Dataset.from_generator(
        generator=my_generator(train_embedding_batch, train_decoder_input, train_output),
        output_types=({'Encoder_In': tf.float32, 'Decoder_In': tf.float32}, tf.float32),
        output_shapes=({'Encoder_In': (BB_MAX_NUM, BB_VECTOR_DIM), 'Decoder_In': (COMMENT_MAX_LEN, WORD_EMBEDDING_DIM)},
                       (COMMENT_MAX_LEN, WORD_EMBEDDING_DIM))) \
        .shuffle(buffer_size=8192, reshuffle_each_iteration=True, seed=seed).cache() \
        .repeat() \
        .batch(TRAIN_BATCH_SIZE) \
        .map(combine_en_de_input, num_parallel_calls=32) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        generator=my_generator(val_embedding_batch, val_decoder_input, val_output),
        output_types=({'Encoder_In': tf.float32, 'Decoder_In': tf.float32}, tf.float32),
        output_shapes=({'Encoder_In': (BB_MAX_NUM, BB_VECTOR_DIM), 'Decoder_In': (COMMENT_MAX_LEN, WORD_EMBEDDING_DIM)},
                       (COMMENT_MAX_LEN, WORD_EMBEDDING_DIM))) \
        .shuffle(buffer_size=8192, reshuffle_each_iteration=True, seed=seed).cache() \
        .repeat() \
        .batch(TRAIN_BATCH_SIZE) \
        .map(combine_en_de_input, num_parallel_calls=32) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    print("Complete the data set preparation.")

    if my_model is None and model_load is None:
        my_model = customize_model()
    elif model_load is not None:
        with strategy.scope():
            my_model = tf.keras.models.load_model(path_prefix + 'results/' + model_load + '.model')
        print('Load the Model', model_load)

    with open(path_prefix + 'results/' + timeRecord + '.log', mode='a') as result_f:
        result_f.write('Hyper parameters: ' + PrintConfig())
        if model_load is not None:
            result_f.write('Load the Model: ' + model_load + '\n')

    def my_result(s):
        with open(path_prefix + 'results/' + timeRecord + '.log', mode='a') as result_file:
            print(s, file=result_file)
            result_file.write('\n')

    my_model.summary(print_fn=my_result)
    print("Model input shape: ", my_model.input_shape)
    print("Model output shape: ", my_model.output_shape)

    LR_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor=CALL_BACK_MONITOR, factor=0.5, patience=32,
                                                       min_delta=0.0001, min_lr=MIN_LR, mode='auto')
    Early_stop = tf.keras.callbacks.EarlyStopping(monitor=CALL_BACK_MONITOR, min_delta=0.0001, patience=64, mode='auto')
    CSV_log = tf.keras.callbacks.CSVLogger(path_prefix + 'results/' + timeRecord + '.csv')
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(path_prefix + 'results/' + timeRecord + '.model', verbose=1,
                                                    monitor=CALL_BACK_MONITOR, save_best_only=True, mode='auto')
    callback_list = [LR_callback, Early_stop, CSV_log, Checkpoint]

    hist = my_model.fit(train_dataset, epochs=EPOCHS, steps_per_epoch=STEP_PER_EPOCH,
                        verbose=1, callbacks=callback_list,
                        validation_data=val_dataset, validation_steps=STEP_PER_EPOCH)

    with open(path_prefix + 'results/' + timeRecord + '.log', mode='a') as result_f:
        result_f.write('--------------------------------------THEN TEST------------------------------------------\n')

    plt.plot(hist.history['loss'][1:])
    plt.plot(hist.history['val_loss'][1:])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.tight_layout()
    plt.savefig(path_prefix + 'results/' + timeRecord + '.png')
    plt.clf()
    print('Generated the figure of epoch -> loss')

    plt.close()
    my_model.save(path_prefix + 'results/' + timeRecord + '_Final.model')

    print('--------------------------------------------TEST-------------------------------------------')
    if my_model is None:
        print("ERROR WITH MY_MODEL!")
        return -1

    test_results = np.zeros(shape=(len(test_input), COMMENT_MAX_LEN, WORD_EMBEDDING_DIM))
    test_decoder_input = np.zeros(shape=(len(test_input), COMMENT_MAX_LEN, WORD_EMBEDDING_DIM))
    test_decoder_input = np.insert(test_decoder_input, 0, my_Gensim_word2vec_model.wv.word_vec('<GO>'), axis=1)[:, :, :]
    test_decoder_input = test_decoder_input[:, :-1, :]
    start = time.time()
    for i in range(COMMENT_MAX_LEN):
        ans = my_model.predict([test_input, test_decoder_input])[:, i, :]
        test_results[:, i, :] = ans
        if i + 1 < COMMENT_MAX_LEN:
            test_decoder_input[:, i + 1, :] = ans
    end = time.time()
    print('Prediction time (no EOS to interrupt output:', end - start)

    with open(path_prefix + 'results/' + timeRecord + '.log', mode='a') as result_f:
        result_f.write('Prediction time (no EOS to interrupt output: ' + str(end - start) + ' With ' +
                       str(len(test_results)) + ' samples.' + '\n')
        result_f.write('--------------------------------------INPUT-----------------------------------------\n')
        for i, ele in enumerate(test_input[: 10]):
            result_f.write(str(i) + ': ')
            json.dump(ele.tolist(), result_f)
            result_f.write('\n')
        result_f.write('--------------------------------------OUTPUT-----------------------------------------\n')
        for i, ele in enumerate(test_results[: 10]):
            result_f.write(str(i) + ': ')
            json.dump(ele.tolist(), result_f)
            result_f.write('\n')

    reference = test_comment_batch
    candidate = []

    for i, res in enumerate(test_results):
        predicted_comment = decode_vec2word(res.tolist(), my_Gensim_word2vec_model.wv)
        output_stand = test_comment_batch[i].copy()
        with open(path_prefix + 'results/' + timeRecord + '.log', mode='a') as result_f:
            result_f.write('-------------------------------------------------------------------------------\n')
            result_f.write("The predicted comment: " + predicted_comment + '\n')
            result_f.write("The standard comment: " + ' '.join(output_stand) + '\n')
        candidate.append(predicted_comment.split(' '))

    precise, recall, F1_score, count = 0, 0, 0, 0
    for ref, cand in zip(reference, candidate):
        True_positive, False_positive, False_negative = 0, 0, 0
        for w in cand:
            if w in ref:
                True_positive = True_positive + 1
            else:
                False_positive = False_positive + 1
        for w in ref:
            if w not in cand: False_negative = False_negative + 1

        if True_positive + False_positive != 0:
            temp_precise = True_positive / (True_positive + False_positive)
        else:
            temp_precise = 0

        if True_positive + False_negative != 0:
            temp_recall = True_positive / (True_positive + False_negative)
        else:
            temp_recall = 0

        if temp_precise == 0 and temp_recall == 0:
            temp_F1_score = 0
        else:
            temp_F1_score = (2 * temp_precise * temp_recall) / (temp_precise + temp_recall)

        precise = precise + temp_precise
        recall = recall + temp_recall
        F1_score = F1_score + temp_F1_score
        count = count + 1

    precise = precise / count
    recall = recall / count
    F1_score = (2 * precise * recall) / (precise + recall)
    print("THE PRECISE SCORE: ", precise)
    print("THE RECALL SCORE: ", recall)
    print("THE F1-SCORE SCORE: ", F1_score)
    with open(path_prefix + 'results/' + timeRecord + '.log', mode='a') as result_f:
        result_f.write('------------------------------ Precision, Recall, F1-socre -----------------------------\n')
        result_f.write("THE PRECISE SCORE: " + str(precise) + '\n')
        result_f.write("THE RECALL SCORE: " + str(recall) + '\n')
        result_f.write("THE F1-SCORE SCORE: " + str(F1_score) + '\n')

    tf.keras.backend.clear_session()
    return timeRecord


def test(path, On_Service=False):
    path_prefix = normal_path
    path = path_prefix + 'results/' + path
    if not os.path.exists(path + '.model/'):
        print('Wrong path!!!', path + '.model/')
        return -1
    result = open(path + '.individual_test.result', mode='w')

    all_train_comments, comment_batch, embedding_batch = prepare_data(path_prefix + file_path, only_test=True)
    if not comment_batch or not embedding_batch:
        print("Please prepare the comment.record file and embeddings.record files.")
        return -1
    assert len(comment_batch) == len(embedding_batch)

    if os.path.exists(path + '_word2vec.model'):
        Loaded_Gensim_word2vec_model = gensim.models.Word2Vec.load(path + '_word2vec.model')
    else:
        Loaded_Gensim_word2vec_model = gensim.models.Word2Vec.load(path_prefix + 'results/My_comment_word2vec.model')
    print('Successfully load the word2vec model.')

    Loaded_seq2seq_model = tf.keras.models.load_model(path + '.model')  # , custom_objects=custom_layers
    print('Successfully load the ' + path + ' seq2seq model.')

    if Loaded_seq2seq_model is None:
        print("ERROR WITH MY_MODEL!")
        return -1

    for pack_name in embedding_batch:
        if len(embedding_batch[pack_name]) == 0:
            continue
        test_individual_input = []
        for embeddings in embedding_batch[pack_name]:
            test_individual_input.append(np.array([np.array(emb) for emb in embeddings]))
        assert len(embedding_batch[pack_name]) == len(test_individual_input)
        test_input = np.array([ind for ind in test_individual_input])

        print('--------------------------------------------TEST-------------------------------------------')
        result.write('--------------------------------------------TEST-------------------------------------------\n')

        test_results = np.zeros(shape=(len(test_input), COMMENT_MAX_LEN, WORD_EMBEDDING_DIM))
        test_decoder_input = np.zeros(shape=(len(test_input), COMMENT_MAX_LEN, WORD_EMBEDDING_DIM))
        test_decoder_input = np.insert(test_decoder_input, 0, Loaded_Gensim_word2vec_model.wv.word_vec('<GO>'), axis=1)[
                             :, :, :]
        test_decoder_input = test_decoder_input[:, :-1, :]
        for i in range(COMMENT_MAX_LEN):
            ans = Loaded_seq2seq_model.predict([test_input, test_decoder_input])[:, i, :]
            test_results[:, i, :] = ans
            if i + 1 < COMMENT_MAX_LEN:
                test_decoder_input[:, i + 1, :] = ans

        reference = comment_batch[pack_name]
        candidate = []

        for i, res in enumerate(test_results):
            predicted_comment = decode_vec2word(res.tolist(), Loaded_Gensim_word2vec_model.wv)
            output_stand = reference[i].copy()
            result.write('-------------------------------------------------------------------------------\n')
            result.write("The predicted comment: " + predicted_comment + '\n')
            result.write("The standard comment: " + ' '.join(output_stand) + '\n')
            candidate.append(predicted_comment.split(' '))

        precise, recall, F1_score, count = 0, 0, 0, 0
        for ref, cand in zip(reference, candidate):
            True_positive, False_positive, False_negative = 0, 0, 0
            for w in cand:
                if w in ref:
                    True_positive = True_positive + 1
                else:
                    False_positive = False_positive + 1
            for w in ref:
                if w not in cand: False_negative = False_negative + 1

            if True_positive + False_positive != 0:
                temp_precise = True_positive / (True_positive + False_positive)
            else:
                temp_precise = 0

            if True_positive + False_negative != 0:
                temp_recall = True_positive / (True_positive + False_negative)
            else:
                temp_recall = 0

            if temp_precise == 0 and temp_recall == 0:
                temp_F1_score = 0
            else:
                temp_F1_score = (2 * temp_precise * temp_recall) / (temp_precise + temp_recall)

            precise = precise + temp_precise
            recall = recall + temp_recall
            F1_score = F1_score + temp_F1_score
            count = count + 1

        precise = precise / count
        recall = recall / count
        if precise + recall != 0:
            F1_score = (2 * precise * recall) / (precise + recall)
        else:
            F1_score = 0
        print("FOR PACKAGE:", pack_name)
        print("THE PRECISE SCORE:", precise)
        print("THE RECALL SCORE:", recall)
        print("THE F1-SCORE SCORE:", F1_score)
        result.write('------------------------------ Precision, Recall, F1-socre -----------------------------\n')
        result.write("FOR PACKAGE: " + pack_name + '\n')
        result.write("THE PRECISE SCORE: " + str(precise) + '\n')
        result.write("THE RECALL SCORE: " + str(recall) + '\n')
        result.write("THE F1-SCORE SCORE: " + str(F1_score) + '\n')

    result.close()


if __name__ == '__main__':
    global file_path
    file_path = "data/"

    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--File', '-f', required=False, help='Input the file path of the dataset.')
    args = parser.parse_args()
    if args.File:
        file_path = args.File

    time_stamp = main(Word2vec_train_set='New', On_Service=True)
