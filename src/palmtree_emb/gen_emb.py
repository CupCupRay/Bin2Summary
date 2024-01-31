import os
import sys
import time
from palmtree_utils.config import *
from torch import nn
from scipy.ndimage.filters import gaussian_filter1d
from torch.autograd import Variable
import json
import torch
import random
import numpy as np
import palmtree_utils.eval_utils as eval_utils

sys.path.append("../")
from utils import use_depgraph
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

MAX_DATAFLOW_BLOCK = 100


def generate_embedding(models, com_output, emb_output, my_com, addr_block_dict, fbh_blocks):
    com_record = open(com_output, mode='a')
    emb_record = open(emb_output, mode='a')

    embedding_shape = None
    for model in models:
        com_record.write('%%Function: => ' + my_com + '\n')
        function_embedding = None
        for block in addr_block_dict:
            block_embeddings = model.encode(addr_block_dict[block])
            if int(block, 16) not in fbh_blocks:
                block_embeddings = np.array([block_embeddings.sum(0).tolist()])

            if function_embedding is None:
                function_embedding = block_embeddings
            elif function_embedding is not None:
                function_embedding = np.concatenate((function_embedding, block_embeddings), axis=0)

        embedding_shape = function_embedding.shape
        if function_embedding is not None:
            json.dump(function_embedding.tolist(), emb_record)
            emb_record.write('\n')
    return 0


def pick_com(coms):
    if len(coms) == 1:
        return coms[0]
    elif len(coms) > 1:
        return random.choice(coms)
    elif not coms:
        return ''


def find_related_comment(func_com_dict, func_n):
    possible_comments = []

    pure_func_name_words = ''
    if not func_n.startswith('0x'):
        pure_func_name_words = str(func_n.strip('_'))

    func_n = func_n.strip('_')
    dot_index = func_n[:5].find('.')
    if dot_index != -1:
        func_n = func_n[dot_index + 1:]
    # func_n = func_n.replace('sym.', '')
    # func_n = func_n.replace('dbg.', '')

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
    if pick_com(possible_comments) == '' or pick_com(possible_comments) == '[]':
        return pure_func_name_words"""
    return pick_com(possible_comments)


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


def generate_dataset(my_choice, FBH_FLAG, env_data_path, palmtree_path, bin_path):
    data_path = env_data_path + palmtree_path
    my_model = data_path + "train_transformer"
    my_vocab = data_path + "train_vocab"

    if not os.path.exists(data_path + "cfg_" + my_choice.lower() + "_single_new.txt"):
        print('Error! No', my_choice.lower(), ' file in your folder!')
        return -1

    my_test_file = data_path + "cfg_" + my_choice.lower() + "_single_new.txt"
    com_file_path = data_path + my_choice.lower() + "_comment.record"
    emb_file_path = data_path + my_choice.lower() + "_embeddings.record"

    print('Start to generate dataset:', my_choice)

    model_list = [eval_utils.UsableTransformer(model_path=my_model + '.ep19', vocab_path=my_vocab)]

    com_record = open(com_file_path, mode='w')
    emb_record = open(emb_file_path, mode='w')
    com_record.close()
    emb_record.close()

    funcname_comment = get_funcname_comment_dict(data_path)

    func_blocks = dict()  # <addr, instructions>
    current_ins = []
    binary_name, func_name, my_comment, current_addr = '', '', '', ''

    for line in open(my_test_file, mode='r'):
        line = line.strip('\n')
        if '%%Function: => ' in line:
            if len(list(func_blocks.keys())) > 2 and binary_name != '' and func_name != '':
                my_comment = find_related_comment(funcname_comment, func_name)
                if my_comment == '':
                    print('WARNING!', func_name)

                FBH_Block, FBH_Block_int = [], []
                if FBH_FLAG:
                    start = time.time()
                    if len(func_blocks) < MAX_DATAFLOW_BLOCK:
                        binary_path = binary_name

                        FBH_Block = use_depgraph.depgraph_analysis(binary_path, data_path, func_name)

                    for block in FBH_Block:
                        FBH_Block_int.append(int(block, 16))

                generate_embedding(model_list, com_file_path, emb_file_path, my_comment, func_blocks, FBH_Block_int)
                func_blocks = dict()
            _, func_name = line.split('%%Function: => ', 1)

        elif '%%BinaryFile: => ' in line:
            _, binary_name = line.split('%%BinaryFile: => ', 1)
            print('In binary:', binary_name)
        elif '%%Addr: => ' in line:
            _, addr = line.split('%%Addr: => ', 1)
            if current_ins and current_addr != '':
                if current_addr not in func_blocks:
                    func_blocks[current_addr] = current_ins
                else:
                    temp_ins = func_blocks[current_addr] + current_ins
                    func_blocks[current_addr] = temp_ins
            current_addr = addr
            current_ins = []
        else:
            current_ins.append(line)

    if len(list(func_blocks.keys())) > 2 and binary_name != '' and func_name != '':
        my_comment = find_related_comment(funcname_comment, func_name)
        if my_comment == '':
            print('WARNING!', func_name)

        FBH_Block, FBH_Block_int = [], []
        if FBH_FLAG:
            start = time.time()
            if len(func_blocks) < MAX_DATAFLOW_BLOCK:
                binary_path = binary_name

                FBH_Block = use_depgraph.depgraph_analysis(binary_path, data_path, func_name)
            for block in FBH_Block:
                FBH_Block_int.append(int(block, 16))

        generate_embedding(model_list, com_file_path, emb_file_path, my_comment, func_blocks, FBH_Block_int)


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--FSE', '-f', required=False, action='store_true', help='Enable the FSE module.')
    args = parser.parse_args()
    FSE_FLAG = args.FSE

    bin_folder = '../../binaries/'
    local_path = "../../"
    palmtree_path = 'data/'

    if FSE_FLAG:
        print('Enable the FSE mode.')
    else: print('Disable all the module. Use basic version of PalmTree.')

    generate_dataset('train', FSE_FLAG, local_path, palmtree_path, bin_folder)
    generate_dataset('val', FSE_FLAG, local_path, palmtree_path, bin_folder)
    generate_dataset('test', FSE_FLAG, local_path, palmtree_path, bin_folder)
