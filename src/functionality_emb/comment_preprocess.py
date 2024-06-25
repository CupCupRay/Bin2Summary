import os
import nltk
import json
import time
import string
import stanfordcorenlp
from nltk.corpus import wordnet
from nltk.parse import corenlp

# -------------------------------------------------IMPORTANT-------------------------------------------------
# Outside RUN => java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# -------------------------------------------------IMPORTANT-------------------------------------------------

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


STANFORD = "./stanford-corenlp-4.3.0"
# nlp_handler = stanfordcorenlp.StanfordCoreNLP(STANFORD, lang='en')
nlp_handler = stanfordcorenlp.StanfordCoreNLP('http://localhost:9000/', port=9000)
probs = {'annotators': 'pos,lemma',
         'pipelineLanguage': 'en',
         'outputFormat': 'json'}


def GetSyn(word):
    synonyms = []
    for i, syn in enumerate(wordnet.synsets(word)):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms


def WordTokenize(sentence):
    word_list = nltk.tokenize.word_tokenize(sentence)
    return word_list


def LemmatizeAll(words):
    Result = []
    wnl = nltk.stem.WordNetLemmatizer()
    for word, tag in nltk.pos_tag(words):
        if tag.startswith('NN'):
            Result.append(wnl.lemmatize(word, pos='n'))
        elif tag.startswith('VB'):
            Result.append(wnl.lemmatize(word, pos='v'))
        elif tag.startswith('JJ'):
            Result.append(wnl.lemmatize(word, pos='a'))
        elif tag.startswith('R'):
            Result.append(wnl.lemmatize(word, pos='r'))
        else: Result.append(word)
    return Result


def ObtainCom(line):
    line = line.replace('\n', '').replace('\r', '')
    func_name = line[line.find('Name:') + 5: line.find('Comment:')]
    func_name = func_name.strip(' ').strip(';')
    real_com = line[line.find('Comment: ') + 8:]
    while real_com.startswith(' '): real_com = real_com[1:]
    while real_com.endswith(' '): real_com = real_com[:-1]
    comment = real_com
    assert type(comment) == str

    for pun in string.punctuation:
        while len(comment) > 1 and comment.startswith(pun): comment = comment[1:]
    comment = comment.replace('%', '')
    comment = comment.replace('. ', '%')
    comment = comment.replace('.', '')
    comment = comment.replace('%', '. ')
    comment = comment + '.'
    return func_name, comment


def CleanCom(sentence):
    for pun in string.punctuation:
        sentence = sentence.replace(pun, '_')

    words = sentence.split(' ')
    new_words = []
    for word in words:
        while word.endswith('_'): word = word[:-1]
        while word.startswith('_'): word = word[1:]
        if word: new_words.append(word)
    while '' in new_words: new_words.remove('')
    function_words = ['a', 'an', 'these', 'those', 'this', 'that', 'the', 'there']
    for w in function_words:
        if w in new_words: new_words.remove(w)
    return new_words


Main_relation = ['nsubj', 'obj', 'iobj', 'csubj', 'compound', 'dep', 'ccomp', 'xcomp', 'obl', 'conj', 'acl']  # Nominals
Functional_relation = ['aux', 'cop', 'mark', 'clf', 'case', 'nummod', 'cc', 'conj', 'advmod', 'det']  # Function words
def ExtractMainClause(dependency, log=None):
    global Main_relation
    global Functional_relation

    main_token = []
    functional_token = []
    # Add the ROOT
    for arg, tail, head in dependency:
        if arg == 'ROOT' and tail == 0 and head not in main_token:
            main_token.append(head)
            if log is not None: log.write(str(head) + ' ROOT (main)\n')

    last_main_token = []
    last_functional_token = []

    while len(main_token) != len(last_main_token) or len(functional_token) != len(last_functional_token):
        last_main_token = main_token.copy()
        last_functional_token = functional_token.copy()
        for relation, tail, head in dependency:
            Main_MATCH = False
            Functional_MATCH = False
            for standard in Main_relation:
                if (standard in relation and ':' in relation) or standard == relation:
                    Main_MATCH = True
            for standard in Functional_relation:
                if (standard in relation and ':' in relation) or standard == relation:
                    Functional_MATCH = True
            if Main_MATCH and tail in main_token and head not in main_token:
                main_token.append(head)
                if log is not None: log.write('[' + relation + ', ' + str(tail) + ', ' + str(head) + '] '
                                              + str(head) + ' (main)\n')
            elif Functional_MATCH and (tail in main_token or tail in functional_token) and head not in functional_token:
                functional_token.append(head)
                if log is not None: log.write('[' + relation + ', ' + str(tail) + ', ' + str(head) + '] '
                                              + str(head) + ' (functional)\n')
    if log is not None: log.write('\n')
    return list(set(main_token + functional_token))


def PreHandle(com, log=None, Add_subj=False):

    if Add_subj:
        com_words = com.split(' ')
        new_com_words = [com_words[0].lower()] + com_words[1:]
        # Add special subject
        com = 'You ' + ' '.join(new_com_words)
    result_json = nlp_handler.annotate(com, properties=probs)

    result_dict = json.loads(result_json)
    token_index = dict()
    index_pos = dict()
    if not result_dict['sentences']:
        return com, token_index, index_pos
    else: tokens = result_dict['sentences'][0]

    for prob in tokens['tokens']:
        if log is not None:
            log.write(str(prob['index']) + ': ' + str(prob['lemma']) + ', ')
            log.write(str(prob['index']) + ': ' + str(prob['pos']) + '\n')
        token_index[prob['index']] = prob['lemma']
        index_pos[prob['index']] = prob['pos']

    return com, token_index, index_pos


def HandleComment(com, log=None, mode='All', Add_subj=False):
    new_words = []

    com, token_index, index_pos = PreHandle(com, log=log, Add_subj=Add_subj)
    if not token_index or not index_pos:
        return ' '.join(new_words)
    assert type(token_index) == dict
    assert type(index_pos) == dict

    dependency = nlp_handler.dependency_parse(com)
    FLAG = False
    idx = 0
    for relation, tail, head in dependency:
        if 'ROOT' in relation and FLAG:
            break
        elif 'ROOT' in relation:
            FLAG = True
        idx = idx + 1
    dependency = dependency[:idx]
    if log is not None:
        log.write('Dependency parse: ' + str(dependency) + '\n')
    main_clause_idx = ExtractMainClause(dependency, log)

    valid_verb = ['VB', 'VBP', 'VBZ']

    if mode == 'Main':
        Meaningful = False
        for idx in main_clause_idx:
            if idx in token_index:
                if idx in index_pos and (index_pos[idx] in valid_verb or 'NN' in index_pos[idx]):
                    Meaningful = True
                new_words.append(token_index[idx])
        if not Meaningful:
            new_words.clear()

    elif mode == 'Verb':
        for idx in main_clause_idx:
            if idx in index_pos:
                if index_pos[idx] in valid_verb and token_index[idx] != 'be':
                    new_words.append(token_index[idx])
                    break
        assert len(new_words) <= 1

    elif mode == 'Noun':
        for idx in main_clause_idx:
            if idx in index_pos:
                if 'NN' in index_pos[idx]:
                    new_words.append(token_index[idx])
                    break
        assert len(new_words) <= 1

    elif mode == 'All':
        for idx in token_index:
            new_words.append(token_index[idx])
            

    new_com = ' '.join(new_words).lower()
    return new_com


def ShutDown():
    nlp_handler.close()
    print('Close the instance for NLPcore')


def main():
    # VERY IMPORTANT !!!
    COMMENT_MAX_LEN = 20
    PROCESS_MODE = 'Main'  # 'Noun', 'Verb', 'Main'
    path = '../../data/'

    Execute_time = 0
    Execute_num = 0

    with open(path + 'debug.record', mode='w', encoding='utf-8') as debug_file:
        with open(path + '/raw_comment.ref', mode='r') as comment_file, \
                open(path + '/comment.ref', mode='w') as new_comment_file:
            for line in comment_file:
                line = line.strip('\n')
                debug_file.write('----------------------------------------------------------------------------\n')
                data_type, comment = ObtainCom(line)
                debug_file.write(data_type + ' => ' + comment + '\n')
                # debug_file.write(data_type + json.dumps(comment) + '\n')

                # Record the time
                start = time.time()

                new_comment = HandleComment(comment, debug_file, mode=PROCESS_MODE)
                final_comment = CleanCom(new_comment)
                if comment.startswith('fcn_'):
                    final_comment = [comment]
                assert type(final_comment) == list
                if not final_comment:
                    new_comment = HandleComment(comment, debug_file, mode=PROCESS_MODE, Add_subj=True)
                    final_comment = CleanCom(new_comment)
                    assert type(final_comment) == list
                    end = time.time()
                    Execute_time += (end - start)
                    Execute_num += 1

                new_comment_file.write(data_type + ' => ' + json.dumps(final_comment) + '\n')

                debug_file.write(data_type + ' '.join(final_comment) + '\n')
                debug_file.write(data_type + json.dumps(final_comment) + '\n')

    ShutDown()


if __name__ == '__main__':
    main()
