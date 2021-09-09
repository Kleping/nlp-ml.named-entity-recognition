from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import tensorflow as tf
import random as rn
import json
import os

EPOCHS = 50
BATCH_SIZE = 128
LATENT_DIM = 128

tf.keras.activations.softmax()

NUM_WORDS_TRAIN = 0

MODEL_NAME = 'ie'
ROOT = ''
ROOT_DATA = 'data/'
ROOT_MODELS = 'models/'

PATH_DATA_TRAIN = '{}{}train.txt'.format(ROOT, ROOT_DATA)
PATH_DATA_VALID = '{}{}valid.txt'.format(ROOT, ROOT_DATA)

PATH_TOKENIZER_CONFIG_TRAIN = '{}{}tokenizer_train_config.json'.format(ROOT, ROOT_DATA)

PATH_MODEL_DIR = '{}{}'.format(ROOT, ROOT_MODELS)
PATH_MODEL = '{}{}.h5'.format(PATH_MODEL_DIR, MODEL_NAME)
PATH_CHECKPOINT = '{}checkpoint'.format(ROOT)

TOKEN_BEG = 'BOS'
TOKEN_END = 'EOS'
TOKEN_OOV = 'OOV'
TOKEN_PAD = 'PAD'
TOKEN_ORG = 'ORG'
TOKEN_PER = 'PER'
VOC_TARGET = [TOKEN_PAD, TOKEN_OOV, TOKEN_PER, TOKEN_ORG, TOKEN_BEG, TOKEN_END]


def _load_tokenizer_from_json():
    with open(PATH_TOKENIZER_CONFIG_TRAIN) as _f:
        _tokenizer_train = tokenizer_from_json(json.load(_f))

    return _tokenizer_train


def encode_target(text):
    return '{} {} {}'.format(TOKEN_BEG, text, TOKEN_END)


def _take_max_sequence_len(_data_train, _data_valid):
    _max = 0

    for _data in [_data_train, _data_valid]:
        for _sample in _data:
            _len = len(_sample.split('\t')[0].split(' '))
            if _len > _max:
                _max = _len

    return _max, _max + len([TOKEN_BEG, TOKEN_END])


def write_config(_max_source, _max_target):
    _config = {
        'max_source': _max_source,
        'max_target': _max_target,
    }
    with open('{}{}.config'.format(PATH_MODEL_DIR, MODEL_NAME), 'w') as config_file:
        json.dump(_config, config_file, indent=4)


def read_config():
    with open('{}{}.config'.format(PATH_MODEL_DIR, MODEL_NAME), 'r') as config_file:
        _config = json.load(config_file)

    return _config['max_source'], _config['max_target']


def create_model():
    encoder_inputs = tf.keras.Input(shape=(None,))
    encoder_emb = tf.keras.layers.Embedding(NUM_WORDS_TRAIN, LATENT_DIM)(encoder_inputs)

    encoder_0 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_last = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LATENT_DIM * 2,
            return_sequences=True,
            return_state=True,
            dropout=.4,
            recurrent_dropout=.4
        )
    )

    encoder_output_0 = encoder_0(encoder_emb)
    encoder_output_1 = encoder_1(encoder_output_0)
    encoder_output_2 = encoder_2(encoder_output_1)
    encoder_stack, forward_h, forward_c, backward_h, backward_c = encoder_last(encoder_output_2)

    encoder_last_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
    encoder_last_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

    encoder_states = [encoder_last_h, encoder_last_c]

    decoder_inputs = tf.keras.Input(shape=(None,))
    decoder_emb = tf.keras.layers.Embedding(len(VOC_TARGET), LATENT_DIM)(decoder_inputs)

    decoder = tf.keras.layers.LSTM(
        LATENT_DIM * 4,
        return_sequences=True,
        return_state=True,
        dropout=.4,
        recurrent_dropout=.4
    )

    decoder_stack_h, _, _ = decoder(decoder_emb, initial_state=encoder_states)

    context = tf.keras.layers.Attention()([decoder_stack_h, encoder_stack])
    decoder_concat_input = tf.keras.layers.concatenate([context, decoder_stack_h])

    dense = tf.keras.layers.Dense(
        len(VOC_TARGET),
        activation='softmax',
        # kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
        # bias_regularizer=tf.keras.regularizers.l2(1e-4),
        # activity_regularizer=tf.keras.regularizers.l2(1e-5)
    )

    decoder_stack_h = tf.keras.layers.TimeDistributed(dense)(decoder_concat_input)

    return tf.keras.Model([encoder_inputs, decoder_inputs], decoder_stack_h)


def compile_model(m):
    optimizer = tf.keras.optimizers.Adam()

    m.compile(
        optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy']
    )
    return m


class DataSupplier(tf.keras.utils.Sequence):
    def __init__(self, _config, _data):
        self._batch_size = _config['batch_size']
        self._data = _data
        self._tokenizer = _config['tokenizer']
        self._voc_target = _config['voc_target']
        self._max_source = _config['max_source']
        self._max_target = _config['max_target']
        if self._data is not None:
            rn.shuffle(self._data)

    def __len__(self):
        return int(np.floor(len(self._data) / self._batch_size))

    def __getitem__(self, ndx):
        source, target = self.extract_batch(ndx, self._batch_size, self._data)
        return self.encode_data(source, target)

    def on_epoch_end(self):
        rn.shuffle(self._data)

    def encode_data(self, _source_texts, _target_texts):
        encoder_input_data = np.zeros(
            (self._batch_size, self._max_source), dtype='int32'
        )
        decoder_input_data = np.zeros(
            (self._batch_size, self._max_target), dtype='int32'
        )
        decoder_target_data = np.zeros(
            (self._batch_size, self._max_target), dtype='int32'
        )

        # source encoding
        for _seq_num, _seq in enumerate(self._tokenizer.texts_to_sequences(_source_texts)):
            for _token_idx, _token in enumerate(_seq):
                encoder_input_data[_seq_num, _token_idx] = _token

        # DEBUGGING
        # print(_source_texts[0])
        # print(encoder_input_data[0])
        # print()

        # target encoding
        for _text_num, _text in enumerate(_target_texts):
            for _token_idx, _token in enumerate(_text.split(' ')):
                if _token == TOKEN_PER or _token == TOKEN_ORG or _token == TOKEN_BEG or _token == TOKEN_END:
                    _i = VOC_TARGET.index(_token)
                else:
                    _i = VOC_TARGET.index(TOKEN_OOV)

                decoder_input_data[_text_num, _token_idx] = _i

                if _token_idx > 0:
                    decoder_target_data[_text_num, _token_idx - 1] = _i

        # DEBUGGING
        # print(_target_texts[0])
        # print(decoder_input_data[0])
        # print(decoder_target_data[0])
        # exit()

        return [encoder_input_data, decoder_input_data], decoder_target_data

    @staticmethod
    def append_sample(sample, source, target):
        source_item, target_item = sample.split('\t')
        source.append(source_item)
        target.append(encode_target(target_item.strip()))
        return source, target

    def extract_batch(self, idx, _batch_size, _data):
        source = []
        target = []
        ndx_from = idx * _batch_size
        ndx_to = min(idx * _batch_size + _batch_size, len(_data))

        for sample in _data[ndx_from: ndx_to]:
            source, target = self.append_sample(sample, source, target)

        if ndx_to % _batch_size != 0:
            for sample in rn.sample(_data[:ndx_from], _batch_size - len(_data) % _batch_size):
                source, target = self.append_sample(sample, source, target)

        return source, target


with open(PATH_DATA_TRAIN, 'r', encoding='utf-8') as f:
    data_train = f.read().split('\n')[:BATCH_SIZE*10]

with open(PATH_DATA_VALID, 'r', encoding='utf-8') as f:
    data_valid = f.read().split('\n')[:BATCH_SIZE*3]

tokenizer_train = _load_tokenizer_from_json()
NUM_WORDS_TRAIN = len(tokenizer_train.word_counts) + 2
# tokenizer_train.num_words = min(20000, len(tokenizer_train.word_counts))

print()
print('word_counts: {}'.format(len(tokenizer_train.word_counts)))
print('len_data_train: {}'.format(len(data_train)))
print('len_data_valid: {}'.format(len(data_valid)))
print()

if os.path.isfile(PATH_CHECKPOINT):
    max_source, max_target = read_config()
    model = create_model()
    model.load_weights(PATH_CHECKPOINT)
    model = compile_model(model)
    model.save(PATH_MODEL)
    exit()
else:
    max_source, max_target = _take_max_sequence_len(data_train, data_valid)
    write_config(max_source, max_target)
    model = compile_model(create_model())

print(model.summary())

data_supplier_config_train = {
    'tokenizer': tokenizer_train,
    'max_source': max_source,
    'max_target': max_target,
    'voc_target': VOC_TARGET,
    'batch_size': BATCH_SIZE,
}

data_supplier_config_valid = {
    'tokenizer': tokenizer_train,
    'max_source': max_source,
    'max_target': max_target,
    'voc_target': VOC_TARGET,
    'batch_size': BATCH_SIZE,
}

train_supplier = DataSupplier(data_supplier_config_train, data_train)
valid_supplier = DataSupplier(data_supplier_config_valid, data_valid)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=PATH_CHECKPOINT,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
)

model.fit(
    train_supplier,
    validation_data=valid_supplier,
    epochs=EPOCHS,
    shuffle=True,
    # callbacks=[model_checkpoint_callback]
)

model.save(PATH_MODEL)
