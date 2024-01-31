import os
import ast
import json
import random
import time
import math
import shutil
import collections
import numpy as np
import gen_feature
import tensorflow as tf
from deepwalk import deepwalk
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# from binary_analysis import Node

# IMPORTANT !!! numpy scipy gensim==3.8.3
## IMPORTANT!!! Tensorflow-gpu==1.14
DATA_EXPANDING = 5
LEAST_INS = 10

# TF-2
# print('Using the Tensorflow version', tf.version)
# print('GPU:', tf.config.list_physical_devices('GPU'))
# TF-1
print('Using the Tensorflow version', tf.__version__)
print('GPU:', tf.test.is_gpu_available())
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync)


def PrepareInfo(path, info, source_type):
    blockIdxToTokens = {}
    blockIdxToOpcodeNum = {}
    blockIdxToOpcodeCounts = {}
    insToBlockCounts = {}
    blockToIdx = {}
    opcodeList = []
    READ_FLAG = False
    for line in open(path + 'blockIdxToTokens', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if source_type in line and info == line[line.find(' ') + 1:]:
            READ_FLAG = True
        elif source_type in line and info != line[line.find(' ') + 1:]:
            READ_FLAG = False
        elif READ_FLAG:
            index, contents = line.split(': ', 1)
            tokens = contents.split(', ')
            blockIdxToTokens.update({index: tokens})

    READ_FLAG = False
    for line in open(path + 'blockIdxToOpcodeNum', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if source_type in line and info == line[line.find(' ') + 1:]:
            READ_FLAG = True
        elif source_type in line and info != line[line.find(' ') + 1:]:
            READ_FLAG = False
        elif READ_FLAG:
            index, content = line.split(': ', 1)
            opcodenum = int(content)
            blockIdxToOpcodeNum.update({index: opcodenum})

    READ_FLAG = False
    for line in open(path + 'blockIdxToOpcodeCounts', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if source_type in line and info == line[line.find(' ') + 1:]:
            READ_FLAG = True
        elif source_type in line and info != line[line.find(' ') + 1:]:
            READ_FLAG = False
        elif READ_FLAG:
            index, contents = line.split(': ', 1)
            opcodecount = ast.literal_eval(contents)
            blockIdxToOpcodeCounts.update({index: opcodecount})

    READ_FLAG = False
    for line in open(path + 'insToBlockCounts', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if source_type in line and info == line[line.find(' ') + 1:]:
            READ_FLAG = True
        elif source_type in line and info != line[line.find(' ') + 1:]:
            READ_FLAG = False
        elif READ_FLAG:
            index, content = line.split(': ', 1)
            blockcount = int(content)
            insToBlockCounts.update({index: blockcount})

    READ_FLAG = False
    block_count = 0
    for line in open(path + 'blockToIdx', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if source_type in line and info == line[line.find(' ') + 1:]:
            READ_FLAG = True
        elif source_type in line and info != line[line.find(' ') + 1:]:
            READ_FLAG = False
        elif READ_FLAG:
            block_count = block_count + 1
            addr, idx = line.split(' -> ', 1)
            blockToIdx.update({hex(int(addr, 16)): idx})
    print('Number of blocks: ' + str(block_count))

    READ_FLAG = False
    for line in open(path + 'opcodeList', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if source_type in line and info == line[line.find(' ') + 1:]:
            READ_FLAG = True
        elif source_type in line and info != line[line.find(' ') + 1:]:
            READ_FLAG = False
        elif READ_FLAG:
            if ', ' in line:
                opcodes = line.split(', ')
                opcodeList = opcodeList + opcodes
            else:
                opcodeList.append(line)
    opcodeList = set(opcodeList)

    READ_FLAG = False
    edge_count = 0
    with open(path + 'temp_edgeList', mode='w') as new_edgelist:
        for line in open(path + 'edgeList', mode='r'):
            line = line.replace('\r', '').replace('\n', '')
            if source_type in line and info == line[line.find(' ') + 1:]:
                READ_FLAG = True
            elif source_type in line and info != line[line.find(' ') + 1:]:
                READ_FLAG = False
            elif READ_FLAG:
                start, end = line.split(' ')
                if start in blockToIdx and end in blockToIdx:
                    edge_count = edge_count + 1
                    start_index = blockToIdx[start]
                    end_index = blockToIdx[end]
                    # print('Edge: ' + start_index + ' ' + end_index)
                    new_edgelist.write(start_index + ' ' + end_index + '\n')
    new_edgelist.close()
    print('Number of edges: ' + str(edge_count))

    return blockIdxToTokens, \
           blockIdxToOpcodeNum, \
           blockIdxToOpcodeCounts, \
           insToBlockCounts, \
           opcodeList, \
           blockToIdx, \
           edge_count


# Input: blockIdxToTokens: 'index to tokens from each basic block'
# Return: dictionary: 'index to token', reversed_dictionary: 'token to index'
def vocBuild(blockIdxToTokens, opcodelist):
    vocabulary = []
    opcodeidxlist = []
    reversed_dictionary = {}
    count = [['UNK'], -1]
    index = 0
    for idx in blockIdxToTokens:
        # print('In the Block %s' % idx)
        for token in blockIdxToTokens[idx]:
            # print('Token %s' % token)
            vocabulary.append(token)
            if token not in reversed_dictionary:
                reversed_dictionary.update({token: index})
                if token in opcodelist and index not in opcodeidxlist:
                    opcodeidxlist.append(index)
                    print("token:", token, " has idx: ", str(index))
                index = index + 1

    dictionary = dict(zip(reversed_dictionary.values(), reversed_dictionary.keys()))
    count.extend(collections.Counter(vocabulary).most_common(1000 - 1))
    print('20 most common tokens: ', count[:20])

    del vocabulary

    return dictionary, reversed_dictionary, opcodeidxlist


# generate article for word2vec. put all random walks together into one article.
# we put a tag between blocks
def articlesGen(walks, blockIdxToTokens, reversed_dictionary, opcode_idx_list):
    # stores all the articles, each article itself is a list
    article = []

    # stores all the block boundary indice. blockBoundaryIndices[i] is a list to store indices for articles[i].
    # each item stores the index for the last token in the block
    blockBoundaryIdx = []
    test_list = []
    for walk in walks:
        # one random walk is served as one article
        for idx in walk:
            if idx in blockIdxToTokens:
                tokens = blockIdxToTokens[idx]
                for token in tokens:
                    if reversed_dictionary[token] in opcode_idx_list:
                        if len(test_list) > 4:
                            print("WARNING!! " + idx + ': ' + str(test_list))
                        test_list.clear()
                    test_list.append((token, reversed_dictionary[token]))
                    article.append(reversed_dictionary[token])
            blockBoundaryIdx.append(len(article) - 1)
            # aritcle.append(boundaryIdx)

    insnStartingIndices = []
    indexToCurrentInsnsStart = {}
    # blockEnd + 1 so that we can traverse to blockEnd
    # go through the current block to retrive instruction starting indices
    for i in range(0, len(article)):
        if article[i] in opcode_idx_list:
            insnStartingIndices.append(i)
        indexToCurrentInsnsStart[i] = len(insnStartingIndices) - 1

    # for counter, value in enumerate(insnStartingIndices):
    #     if data_index == value:
    #         currentInsnStart = counter
    #         break
    #     elif data_index < value:
    #         currentInsnStart = counter - 1
    #         break

    return article, blockBoundaryIdx, insnStartingIndices, indexToCurrentInsnsStart


# adopt TF-IDF method during block embedding calculation
def CalBlockEmbeddings(blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts,
                       tokenEmbeddings, reversed_dictionary, opcode_idx_list):
    block_embeddings = {}
    totalBlockNum = len(blockIdxToOpcodeCounts)

    for bid in blockIdxToTokens:
        tokenlist = blockIdxToTokens[bid]
        opcodeCounts = blockIdxToOpcodeCounts[bid]
        opcodeNum = blockIdxToOpcodeNum[bid]

        opcodeEmbeddings = []
        operandEmbeddings = []

        if len(tokenlist) != 0:
            for token in tokenlist:
                tokenid = reversed_dictionary[token]

                tokenEmbedding = tokenEmbeddings[tokenid]

                if tokenid in opcode_idx_list and token in opcodeCounts:
                    # here we multiple the embedding with its TF-IDF weight if the token is an opcode
                    tf_weight = opcodeCounts[token] / opcodeNum
                    x = totalBlockNum / insToBlockCounts[token]
                    idf_weight = math.log(x)
                    tf_idf_weight = tf_weight * idf_weight

                    opcodeEmbeddings.append(tokenEmbedding * tf_idf_weight)
                else:
                    operandEmbeddings.append(tokenEmbedding)

            opcodeEmbeddings = np.array(opcodeEmbeddings)
            operandEmbeddings = np.array(operandEmbeddings)

            opcode_embed = opcodeEmbeddings.sum(0)
            operand_embed = operandEmbeddings.sum(0)
        # set feature vector for null block node to be zeros
        else:
            embedding_size = 64
            opcode_embed = np.zeros(embedding_size)
            operand_embed = np.zeros(embedding_size)

        # if no operand, give zeros
        if operand_embed.size == 1:
            operand_embed = np.zeros(len(opcode_embed))

        block_embed = np.concatenate((opcode_embed, operand_embed), axis=0)
        block_embeddings[bid] = block_embed
        # print("bid", bid, "block embedding:", block_embed)

    return block_embeddings


# adopt TF-IDF method during block embedding calculation
def CalInsEmbeddings(blockIdxToTokens, blockIdxToOpcodeNum, blockIdxToOpcodeCounts, insToBlockCounts,
                       tokenEmbeddings, reversed_dictionary, opcode_idx_list):
    ins_embeddings = {}
    totalBlockNum = len(blockIdxToOpcodeCounts)

    for bid in blockIdxToTokens:
        tokenlist = blockIdxToTokens[bid]
        opcodeCounts = blockIdxToOpcodeCounts[bid]
        opcodeNum = blockIdxToOpcodeNum[bid]

        # opcodeEmbeddings = []
        # operandEmbeddings = []
        current_opcodeEmb = []
        opcode_count = 0
        embedding_size = 64
        my_opcode_emb = dict()
        my_operand_emb = dict()

        if len(tokenlist) != 0:
            # print(tokenlist)
            for token in tokenlist:
                tokenid = reversed_dictionary[token]

                tokenEmbedding = tokenEmbeddings[tokenid]

                if tokenid in opcode_idx_list and token in opcodeCounts:
                    opcode_count += 1

                    # here we multiple the embedding with its TF-IDF weight if the token is an opcode
                    tf_weight = opcodeCounts[token] / opcodeNum
                    x = totalBlockNum / insToBlockCounts[token]
                    idf_weight = math.log(x)
                    tf_idf_weight = tf_weight * idf_weight

                    # opcodeEmbeddings.append(tokenEmbedding * tf_idf_weight)
                    current_opcodeEmb = tokenEmbedding * tf_idf_weight
                    my_opcode_emb[opcode_count] = current_opcodeEmb

                else:
                    # operandEmbeddings.append(tokenEmbedding)
                    if opcode_count in my_operand_emb and my_operand_emb[opcode_count]:
                        temp_operand_emb = my_operand_emb[opcode_count]
                        temp_operand_emb.append(tokenEmbedding)
                        my_operand_emb[opcode_count] = temp_operand_emb
                    else:
                        my_operand_emb[opcode_count] = [tokenEmbedding]

            # opcodeEmbeddings = np.array(opcodeEmbeddings)
            # operandEmbeddings = np.array(operandEmbeddings)

            opcodeEmbeddings = []
            operandEmbeddings = []
            for opc in my_opcode_emb:
                opcodeEmbeddings.append(np.array(my_opcode_emb[opc]))
                if opc in my_operand_emb and my_operand_emb[opc]:
                    temp_operand_emb = my_operand_emb[opc]
                else: temp_operand_emb = []
                while len(temp_operand_emb) < 5:
                    temp_operand_emb.append(np.zeros(embedding_size))
                operandEmbeddings.append(np.array(temp_operand_emb))
            opcodeEmbeddings = np.array(opcodeEmbeddings)
            operandEmbeddings = np.array(operandEmbeddings)

            # print('opcodeEmbeddings.shape:', opcodeEmbeddings.shape)
            # print('operandEmbeddings.shape:', operandEmbeddings.shape)
            opcode_embed = opcodeEmbeddings
            operand_embed = operandEmbeddings.sum(1)
            assert opcode_embed.shape == operand_embed.shape
            # print('Final opcodeEmbeddings.shape:', opcode_embed.shape)
            # print('Final operandEmbeddings.shape:', operand_embed.shape)
        # set feature vector for null block node to be zeros
        else:
            opcode_embed = np.zeros(embedding_size)
            operand_embed = np.zeros(embedding_size)
            print('ZEROS: opcode_embed.shape:', opcode_embed.shape)
            print('ZEROS: operand_embed.shape:', operand_embed.shape)

        # if no operand, give zeros
        # if operand_embed.size == 1:
            # operand_embed = np.zeros(len(opcode_embed))

        block_embed = np.concatenate((opcode_embed, operand_embed), axis=1)
        ins_embeddings[bid] = block_embed
        # print("bid", bid, "block embedding:", block_embed)

    return ins_embeddings


def InsVecFileGen(feature_file, block_embeddings):
    with open(feature_file, mode='a') as feaVecFile:
        counter = 0
        for key in block_embeddings:
            value = block_embeddings[key]
            # print('value.shape:', value.shape)
            # index as the first element and then output all the features
            for line in range(len(value)):
                single_value = value[line]
                # print('single_value.shape:', single_value.shape)
                feaVecFile.write(str(counter) + " ")
                assert len(single_value) == 128
                for k in range(len(single_value)):
                    feaVecFile.write(str(single_value[k]) + " ")
                feaVecFile.write("\n")
                counter = counter + 1


def FeatureVecFileGen(feature_file, block_embeddings):
    with open(feature_file, mode='a') as feaVecFile:
        for counter in block_embeddings:
            value = block_embeddings[counter]
            # index as the first element and then output all the features
            feaVecFile.write(str(counter) + " ")
            for k in range(len(value)):
                feaVecFile.write(str(value[k]) + " ")
            feaVecFile.write("\n")


def get_funcname_comment_dict(path):
    if not path.endswith('/'): path = path + '/'
    func_comment = dict()
    for line in open(path + 'comment.ref', mode='r'):
        line = line.strip('\n')
        elements = line.split(' => ')
        if len(elements) != 2:
            continue
        funcname = elements[0] + '_'
        comment = elements[1]
        if funcname not in func_comment:
            func_comment[funcname] = comment
        else:
            idx = 0
            funcname = funcname + str(idx)
            while funcname in func_comment:
                idx += 1
                funcname = funcname[:funcname.rfind('_')] + '_' + str(idx)
            func_comment[funcname] = comment
    return func_comment


def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma


def pick_com(coms):
    if len(coms) == 1:
        return coms[0]
    elif len(coms) > 1:
        return random.choice(coms)
    elif not coms:
        return ''


def find_related_comment(func_com_dict, func_n):
    possible_comments = []
    pure_func_name_words = str(func_n.strip('_'))
    func_n = func_n.replace('sym.', '')

    # Exact match
    if func_n in func_com_dict and func_com_dict[func_n] != '[]':
        return func_com_dict[func_n]

    # Partial match
    for this_f in func_com_dict:
        if this_f.startswith(func_n + '_') and func_com_dict[this_f] != '[]':
            possible_comments.append(func_com_dict[this_f])
    if len(possible_comments) == 0:
        for this_f in func_com_dict:
            if func_n in this_f and func_com_dict[this_f] != '[]':
                possible_comments.append(func_com_dict[this_f])

    """
    # Simply use function name
    if pick_com(possible_comments) == '' or pick_com(possible_comments) == []:
        return pure_func_name_words"""
    return pick_com(possible_comments)


def generate_features(filepath, filename, funcname_comment_dict):
    if not os.path.exists(filepath + filename + '.snippet'):
        filename = filename + '.exe'

    for line in open(filepath + filename + '.snippet', mode='r'):
        contents = line.split(' ')
        comment_source = contents[0]

        print('######################################################################################################')
        semantic_info = contents[1]
        temp_semantic_info = semantic_info.strip('_')
        print('Function:', temp_semantic_info)

        my_comment = find_related_comment(funcname_comment_dict, temp_semantic_info)
        if my_comment == '' or my_comment == '[]':
            print('Function does not have comment.')
            continue

        # Step 1: Collect the assembly information for the binary file
        source_type = '%%Function: '
        BIdxToTokens, BIdxToOpcodeNum, BIdxToOpcodeCounts, IToBlockCounts, OList, BToIdx, ECount = \
            PrepareInfo(filepath, semantic_info, source_type)
        if BIdxToTokens == {} or (len(BToIdx) < 2 and list(BIdxToOpcodeNum.values())[0] < LEAST_INS):
            print('This Function has only little instructions, SO WE SKIP THIS FUNCTION')
            continue
        print('Finish the Step 1')

        # Step 2: Vocabulary buildup
        TokenDict, Rever_TokenDict, OIdxList = vocBuild(BIdxToTokens, OList)
        print('Finish the Step 2')

        LOOP = DATA_EXPANDING
        for _ in range(LOOP):
            processed_comment = my_comment

            # Step 3: Generate random walks, each walk contains certain blocks
            print(filepath + "temp_edgeList")
            if len(BIdxToTokens) >= 20:
                Walks = deepwalk.randomWalksGen(filepath + "temp_edgeList", BIdxToTokens, number_walks=2)
            elif len(BIdxToTokens) >= 10:
                Walks = deepwalk.randomWalksGen(filepath + "temp_edgeList", BIdxToTokens, number_walks=8)
            else:
                Walks = deepwalk.randomWalksGen(filepath + "temp_edgeList", BIdxToTokens, number_walks=32)
            # print("Random walks: " + str(Walks))
            print('Finish the Step 3')

            # step 4: Generate articles based on random walks
            Article, BBoundaryIdx, IStartingIndices, IdxToInsStart = \
                articlesGen(Walks, BIdxToTokens, Rever_TokenDict, OIdxList)
            # print("Length of article: " + str(len(Article)))
            if len(Article) <= 0:
                print('This Function has no article, SO WE SKIP THIS FUNCTION')
                continue
            print('Finish the Step 4')

            # step 5: Token embedding generation
            # with strategy.scope():
            TokenEmbeddings = gen_feature.tokenEmbeddingGeneration(Article, BBoundaryIdx, IStartingIndices,
                                                                   IdxToInsStart, TokenDict, Rever_TokenDict, OIdxList)
            print('Finish the Step 5')

            # step 6: Calculate feature vector for blocks
            BlockEmbeddings = CalBlockEmbeddings(BIdxToTokens, BIdxToOpcodeNum, BIdxToOpcodeCounts,
                                                 IToBlockCounts, TokenEmbeddings, Rever_TokenDict, OIdxList)
            FeatureVecFileGen(filepath + filename + '.feature', BlockEmbeddings)
            print('Finish the Step 6')


def prepare_embedding_file(filepath, filename):
    # identifier = filename[:filename.rfind('_')]
    file_emb = []  # Temp Embeddings list of a single comment, list [block_1 <list>, block_2 <list>, ...]
    print("Begin to extract the features (been preprocessed).")
    with open(filepath + filename + '_comment.record', mode='w') as com_record, \
            open(filepath + filename + '_embeddings.record', mode='w') as emb_record:
        for root, dirs, files in os.walk(filepath):
            if '\\' in root: root = root.replace('\\', '/')
            if not root.endswith('/'): root = root + '/'

            for file in files:
                if not file.endswith('.feature'): continue
                print('Open ' + root + file)
                file_emb.clear()
                block_count = 0
                current_com = ''
                com_type = ''
                for line in open(root + file, mode='r'):
                    line = line.replace('\n', '').replace('\r', '')
                    if 'With %%Comment (Processed): ' in line:
                        comment = line[line.find('%%Comment (Processed): ') + len('%%Comment (Processed): '):]
                        com_type = line[: line.find('With %%Comment (Processed): ')]
                        if file_emb and current_com != '':
                            json.dump(file_emb, emb_record)
                            com_record.write(str(com_type) + ' => ' + current_com)
                            # json.dump(current_com, com_record)
                            emb_record.write('\n')
                            com_record.write('\n')
                        current_com = comment
                        file_emb.clear()
                        block_count = 0
                    elif '%%Comment' not in line and line != '':
                        contents = line.split(' ')
                        assert block_count == int(contents[0])
                        embeddings = contents[1:]
                        while '' in embeddings: embeddings.remove('')
                        embeddings2 = standardization(np.array(list(map(float, embeddings))))
                        # embeddings2 = np.array(list(map(float, embeddings)))
                        file_emb.append(embeddings2.tolist())
                        block_count = block_count + 1
                if file_emb and current_com != '':
                    json.dump(file_emb, emb_record)
                    com_record.write(str(com_type) + ' => ' + current_com)
                    # json.dump(, com_record)
                    emb_record.write('\n')
                    com_record.write('\n')
    return 0


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--Filepath', required=True, help='Input a particular data directory')
    args = parser.parse_args()
    Filepath = args.Filepath
    Filepath = Filepath.replace('\\\\', '/')
    if not Filepath.endswith('/'):
        Filepath = Filepath + '/'

    if os.path.exists(Filepath):
        Filename = Filepath[:-1]
        funcname_comment = get_funcname_comment_dict(Filename[:Filename.rfind('/')])
        Filename = Filename[Filename.rfind('/') + 1:]

        generate_features(Filepath, Filename, funcname_comment)
        prepare_embedding_file(Filepath, Filename)

