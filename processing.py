import os
import copy
import itertools
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json, text_to_word_sequence
import stanza
import spacy_stanza
import json
import sys
import regex

stanza.download('ru')
nlp = spacy_stanza.load_pipeline('ru')

T_EOL = 'EOL'
T_ORG = 'ORG'
T_PER = 'PER'
T_OUT = 'OUT'
T_OOV = 'OOV'

FILTERS = r"-<>\[\]()#•@—№*“…”–%+’&/»«.\",!?:;"
LOWER = False
SPLIT = ' '

BATCH_SIZE = 64
TRAIN_CHUNKS = 700 * 4
VALID_CHUNKS = 300 * 4
counter = 0


def extract_text(_l):
    _text = _l[9:-1]
    _text = _text.replace(u'\xa0', ' ')
    _text = _text.replace(u'\u200a', ' ')
    _text = _text.replace(u'\u202f', ' ')
    _text = _text.replace(u'\u2009', ' ')
    _text = _text.replace(u'\u2028', ' ')
    _text = _text.replace(u'\u2002', ' ')
    _text = _text.replace(u'\u3000', ' ')
    _text = _text.replace(u'\u2003', ' ')
    _text = _text.replace(u'\x1f', ' ')
    _text = _text.replace('\t', ' ')
    return _text


def lemmatize_text(_text):
    _lemmatized_text = _text
    _doc = nlp(_text)
    for _token in _doc:
        _lemmatized_text = _lemmatized_text.replace(str(_token), _token.lemma_)
    return _lemmatized_text


if os.path.isfile('data/tokenizer_train_config.json'):
    with open('data/tokenizer_train_config.json') as _f:
        tokenizer_train = tokenizer_from_json(json.load(_f))
else:
    tokenizer_train = Tokenizer(lower=LOWER, oov_token=T_OOV, filters=FILTERS)

    with open('data/raw/nerus_lenta.conllu', 'r', encoding='utf-8') as f:
        texts = list()
        for line in f:
            if line[:9] == '# text = ':
                text = extract_text(line)
                lemmatized_text = text
                doc = nlp(text)
                for token in doc:
                    lemmatized_text = lemmatized_text.replace(str(token), token.lemma_)

                texts.append(lemmatized_text)
                print('train_lematization: {}/{}'.format(len(texts), TRAIN_CHUNKS * BATCH_SIZE))
                if len(texts) == TRAIN_CHUNKS * BATCH_SIZE:
                    break

        tokenizer_train.fit_on_texts(texts)
        with open('data/tokenizer_train_config.json', 'a+', encoding='utf-8') as _f:
            _f.write(json.dumps(tokenizer_train.to_json(), ensure_ascii=False))

    print('train_word_counts: {}'.format(len(tokenizer_train.word_counts)))

SKIP_LINE_COUNT = 0

if os.path.isfile('data/train.txt'):
    with open('data/train.txt', 'r', encoding='utf-8') as f_train:
        SKIP_LINE_COUNT = len(f_train.readlines())

if os.path.isfile('data/valid.txt'):
    with open('data/valid.txt', 'r', encoding='utf-8') as f_valid:
        SKIP_LINE_COUNT += len(f_valid.readlines())


def _filter_text(_text, _filters, _split):
    if sys.version_info < (3,):
        if isinstance(_text, unicode):  # noqa: F821
            translate_map = {
                ord(c): unicode(_split) for c in _filters  # noqa: F821
            }
            _text = _text.translate(translate_map)
        elif len(_split) == 1:
            translate_map = str.maketrans(_filters, _split * len(_filters))
            _text = _text.translate(translate_map)
        else:
            for c in _filters:
                _text = _text.replace(c, _split)
    else:
        translate_dict = {c: _split for c in _filters}
        translate_map = str.maketrans(translate_dict)
        _text = _text.translate(translate_map)

    return _text


def _text_to_word_sequence(_sentence):
    return text_to_word_sequence(_sentence, lower=LOWER, filters=FILTERS)


def _tokenize(_sentence, _tokenizer):
    reverse_word_index = {v: k for k, v in _tokenizer.word_index.items()}
    word_tokens = [reverse_word_index[_idx] for _idx in _tokenizer.texts_to_sequences([_sentence])[0]]

    return word_tokens


class AnnotatedToken:
    def __init__(self, _token, _tag):
        self.token = _token
        self.tag = _tag


def write_annotation(_sentence, _annotated_sentence):

    if _sentence == '' or _annotated_sentence == '':
        return

    global counter
    _file_name = 'train' if counter < TRAIN_CHUNKS * BATCH_SIZE else 'valid'
    with open('data/{}.txt'.format(_file_name), 'a+', encoding='utf-8') as _f_data:
        if counter != 0 and counter != TRAIN_CHUNKS * BATCH_SIZE:
            _f_data.write('\n')
        _f_data.write(_sentence)
        _f_data.write('\t')
        _f_data.write(_annotated_sentence)
    counter += 1
    print('data_formation: {} of {}'.format(counter, BATCH_SIZE * (TRAIN_CHUNKS + VALID_CHUNKS)))

    if counter == BATCH_SIZE * (TRAIN_CHUNKS + VALID_CHUNKS):
        exit()


def write_data_sample(_text, _annotation):
    global SKIP_LINE_COUNT
    if SKIP_LINE_COUNT != 0:
        global counter
        SKIP_LINE_COUNT -= 1
        counter += 1
        return

    _ann_text = _text_to_word_sequence(_text)
    _ann_idx = 0

    for _token_idx, _token in enumerate(_ann_text):
        _ann_token = ''
        _ann_tag = ''
        while _token != _ann_token:
            try:
                _ann_token += _filter_text(_annotation[_ann_idx].token, FILTERS, SPLIT)
            except:
                print(_text)
                print(_token)
                print(_ann_text)
                print(_ann_token)
                exit()
            _ann_tag = _annotation[_ann_idx].tag
            _ann_idx += 1

        if _ann_tag == 'PER' or _ann_tag == 'ORG':
            _ann_text[_token_idx] = _ann_tag

    _sample_to_write = lemmatize_text(' '.join(_text_to_word_sequence(_text)))
    _annotated_sample_to_write = lemmatize_text(' '.join(_ann_text))
    write_annotation(_sample_to_write, _annotated_sample_to_write)


with open('data/raw/nerus_lenta.conllu', 'r', encoding='utf-8') as f:
    text = None
    for line in f:
        if line[:9] == '# text = ':

            if text is not None and regex.search(r'\p{IsCyrillic}', text):
                write_data_sample(text, annotation)

            text = extract_text(line)
            annotation = list()

        elif text is not None and len(line.split()) > 0 and line.split()[0].isdigit():
            columns = line.split('\t')
            tag = columns[-1][6:-1]

            for token in _text_to_word_sequence(columns[1]):
                annotation.append(AnnotatedToken(token, tag))

    if text is not None and regex.search(r'\p{IsCyrillic}', text):
        write_data_sample(text, annotation)

exit()

with open('data/raw/train_sentences.txt', 'r', encoding='utf-8') as f:
    train_sentences = f.read().split('\n')

with open('data/raw/train_nes.txt', 'r', encoding='utf-8') as f:
    train_nes = f.read().split('\n')

with open('data/raw/train_sentences_enhanced.txt', 'r', encoding='utf-8') as f:
    train_sentences_enhanced = f.read().split('\n')

out = {}


def _inject_entities(_tokenized_sentence, _entity_word_form, _entity_token):
    entity_words = _tokenize(_entity_word_form)
    _out_tokenized_sentence = copy.copy(_tokenized_sentence)

    if len(entity_words) == 1:
        entity_w = entity_words[0]
        sentence_entity_indexes = [
            __idx for (__idx, sentence_w) in enumerate(_tokenized_sentence)
            if sentence_w == entity_w
        ]

        for __idx in sentence_entity_indexes:
            _out_tokenized_sentence[__idx] = _entity_token

    else:
        _idx_start = -1
        for (__idx, sentence_w) in enumerate(_tokenized_sentence):

            if _idx_start == -1:
                if sentence_w == entity_words[0]:
                    _idx_start = __idx
            elif __idx - _idx_start == len(entity_words):
                for __idx1 in range(_idx_start, __idx):
                    _out_tokenized_sentence[__idx1] = _entity_token
                _idx_start = -1
            elif _idx_start != -1:
                if sentence_w != entity_words[__idx-_idx_start]:
                    _idx_start = -1

    return _out_tokenized_sentence


for directory in ['devset', 'testset']:
    for root, dirs, files in os.walk('data/raw/factRuEval-2016/{}'.format(directory)):
        for sub in set([f.split('.')[0] for f in files]):
            with open('data/raw/factRuEval-2016/{}/{}.txt'.format(directory, sub), 'r', encoding='utf-8') as f:
                text = f.read()
                sentences = list(
                    itertools.chain(*list(map(
                        lambda s: nltk.sent_tokenize(s, language='russian'),
                        text.replace('\t', '').split('\n')
                    )))
                )

            entities = {}
            with open('data/raw/factRuEval-2016/{}/{}.spans'.format(directory, sub), 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    values = line.split(' ')
                    entity = values[1]
                    if entity == 'org_name' or entity == 'org_descr':
                        token = T_ORG
                    elif entity == 'name' or entity == 'surname' or entity == 'nickname' or entity == 'patronymic':
                        token = T_PER
                    else:
                        continue
                    word_form = line.split('#')[-1].strip().split(' ')
                    word_form = ' '.join(word_form[int(len(word_form)/2):])
                    entities[word_form] = token

            sorted_entities = {
                k: v for k, v in sorted(entities.items(), key=lambda kv: len(_tokenize(kv[0])), reverse=True)
            }

            for sentence in sentences:
                tokenized_sentence = _tokenize(sentence)
                for entity_word_form in sorted_entities.keys():
                    entity_token = entities[entity_word_form]
                    tokenized_sentence = _inject_entities(tokenized_sentence, entity_word_form, entity_token)

                out[sentence] = ' '.join(tokenized_sentence)


for root, dirs, files in os.walk('data/raw/collections'):

    for file_name in set(map(lambda file: file[:-4], files)):
        file_txt = open('{}/{}.txt'.format(root, file_name), 'r', encoding='utf-8')
        file_ann = open('{}/{}.ann'.format(root, file_name), 'r', encoding='utf-8')

        content_txt = file_txt.read()

        sentences = list(
            itertools.chain(*list(map(
                lambda s: nltk.sent_tokenize(s, language='russian'),
                content_txt.replace('\t', '').replace(';', '\n').split('\n')
            )))
        )

        annotated_sentences = sentences.copy()
        annotations = file_ann.readlines()
        annotations.reverse()

        for idx, sentence in enumerate(annotated_sentences):
            tokenized_sentence = _tokenize(sentence)

            for annotation in annotations:
                info = annotation.split('\t')
                role = info[1].split(' ')
                role_name = role[0]

                if role_name == 'PER':
                    entity_token = T_PER
                elif role_name == 'ORG':
                    entity_token = T_ORG
                else:
                    continue

                entity_word_form = info[2][:-1].strip()
                annotated_sentences[idx] = _inject_entities(tokenized_sentence, entity_word_form, entity_token)

        for idx, sentence in enumerate(sentences):
            out[sentence] = ' '.join(annotated_sentences[idx])

        file_txt.close()
        file_ann.close()


for idx, sentence in enumerate(train_sentences_enhanced[:20]):
    words_sentence = _tokenize(sentence)

    for word_idx, word in enumerate(words_sentence):
        if T_ORG in words_sentence[word_idx]:
            words_sentence[word_idx] = T_ORG
        elif T_PER in words_sentence[word_idx]:
            words_sentence[word_idx] = T_PER

    sentence = sentence.replace('{PERSON}', '').replace('{ORG}', '')
    out[sentence] = ' '.join(words_sentence)


for idx, sentence in enumerate(train_sentences):
    markup = train_nes[idx].split(' ')
    tokenized_sentence = _tokenize(sentence)

    while len(markup) != 0:

        if markup[0] == T_EOL:
            markup = []
            continue

        token_info = markup[:3]
        start_idx = int(token_info[0])
        word_len = int(token_info[1])
        token = token_info[2]

        entity_word_form = sentence[start_idx: start_idx + word_len]
        tokenized_sentence = _inject_entities(tokenized_sentence, entity_word_form, token)

        markup = markup[3:]

    out[sentence] = ' '.join(tokenized_sentence)

with open('data/train.txt', 'w', encoding='utf-8') as f:
    lines = []
    for k, v in out.items():
        lines.append(('' if len(lines) == 0 else '\n') + '{}\t{}'.format(k, v))
    f.writelines(lines)
