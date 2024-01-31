import os
import sys
import json
import random
import numpy as np
import asm2vec.asm
import asm2vec.parse
import asm2vec.model
# from asm2vec.parse import parse_fp
# from asm2vec.model import Asm2Vec
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

sys.path.append("../")
from utils import use_depgraph

################################################### IMPORTANT ###################################################
# If you want to test FSE module with asm2vec, please run Deepbindiff version to prepare .instruction files
#################################################################################################################

LEAST_INS_NUM = 10
MAX_DATAFLOW_BLOCK = 100

def simplify_asm(ro, fi, i_path, a_path, func_call=None, mode='write'):
    funcs = list()
    collected_func_call = dict()
    t_path = ro.replace(i_path, a_path)
    if not os.path.exists(t_path):
        os.mkdir(t_path)

    if mode == 'write':
        print("Generating", t_path + '/' + fi.replace('.instructions', '.s'))
        asm_out = open(t_path + '/' + fi.replace('.instructions', '.s'), mode='w')

    # Open the ".instructions" file
    NEXT_BLOCK, FIRST_BLOCK = False, False
    func_name = ''
    for line in open(ro + '/' + fi, mode='r'):
        ins = line.replace('\r', '').replace('\n', '')
        pre_index = ins.find('0x')

        index = 44
        if len(ins) >= index:
            while ins[index - 1] != ' ':
                index = index - 1

        if ins.startswith('/ '):
            # Start of the function
            pre_i = ins.find(': ') + 2
            post_i = ins.find(' (')
            func_name = ins[pre_i: post_i].strip(' ')
            func_name = func_name[func_name.rfind(' ') + 1:]
            func_name = func_name.strip('*').lower()
            funcs.append(func_name)
            if mode == 'write':
                asm_out.write(func_name + ':\n')
            elif mode == 'prepare':
                collected_func_call[func_name] = []

            FIRST_BLOCK = True

        elif ('; arg' in ins or '; var' in ins) and ' @ ' in ins:
            # Variables definition e.g., arg int64_t arg1 @ rdi
            continue

        elif pre_index == 12 and '      ' in ins[pre_index:] and len(ins) > index:
            # Normal instructions
            addr = ins[pre_index:ins.find(' ', pre_index)]
            if NEXT_BLOCK:
                if mode == 'prepare':
                    temp_ins = collected_func_call[func_name]
                    temp_ins.append(addr + ':')
                    collected_func_call[func_name] = temp_ins
                elif mode == 'write' and not FIRST_BLOCK:
                    asm_out.write(addr + ':\n')

                NEXT_BLOCK = False
                FIRST_BLOCK = False

            post_index = ins.find(';', index)
            if post_index == -1: individual_ins = ins[index:]
            else: individual_ins = ins[index: post_index]

            if mode == 'write' and func_call is not None:
                if individual_ins.startswith('call '):
                    parts = individual_ins.split(' ')
                    callee = parts[-1]
                    if callee in func_call:
                        for each_ins in func_call[callee]:
                            asm_out.write(each_ins + '\n')
                        # asm_out.write(addr)

            elif mode == 'write' and func_call is None:
                asm_out.write('      ' + individual_ins + '\n')
            elif mode == 'prepare':
                temp_ins = collected_func_call[func_name]
                temp_ins.append('      ' + individual_ins)
                collected_func_call[func_name] = temp_ins

        elif '; CODE XREF' in ins or '; CALL XREF' in ins or '; DATA XREF' in ins or ins == '':
            # Find the new block at assembly level
            NEXT_BLOCK = True

    if mode == 'write':
        asm_out.close()
    return funcs, collected_func_call


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


def pick_com(coms):
    if len(coms) == 1:
        return coms[0]
    elif len(coms) > 1:
        return random.choice(coms)
    elif not coms: return ''

def find_related_comment(func_com_dict, func_n):
    possible_comments = []
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

    return pick_com(possible_comments)


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--FSE', '-f', required=False, action='store_true', help='Enable the FSE mode.')
    args = parser.parse_args()
    FSE_FLAG = args.FSE

    data_path = '../../data/'

    train_functions = []
    test_functions = dict()
    all_func_ins = dict()
    pack_file_num = dict()
    funcname_comment = get_funcname_comment_dict(data_path)
    for root, dirs, files in os.walk(data_path):
        root = root.replace('\\', '/')

        for file in files:
            if file.endswith('.instructions'):
                pack_name = file[:file.find('_')]
                if pack_name not in pack_file_num:
                    pack_file_num[pack_name] = 1
                else:
                    pack_file_num[pack_name] = pack_file_num[pack_name] + 1

        for file in files:
            if file.endswith('.instructions'):
                _, function_call = simplify_asm(root, file, data_path, data_path, mode='prepare')
                new_function_call = dict()
                for ele in function_call:
                    if len(function_call[ele]) >= LEAST_INS_NUM:
                        new_function_call[ele] = function_call[ele]

                func_list = []
                if FSE_FLAG:
                    func_list, _ = simplify_asm(root, file, data_path, data_path, func_call=new_function_call)
                elif not FSE_FLAG:
                    func_list, _ = simplify_asm(root, file, data_path, data_path)

                target_path = root.replace(data_path, data_path) + '/' + file.replace('.instructions', '.s')
                my_funcs = asm2vec.parse.parse(target_path, func_names=func_list)

                new_my_func = list()
                for ef in my_funcs:
                    possible_com = find_related_comment(funcname_comment, ef.name())
                    if len(ef) >= LEAST_INS_NUM and possible_com != '' and possible_com != '[]':
                        new_my_func.append(ef)

                train_functions = train_functions + new_my_func
                test_functions[root + '/' + file] = new_my_func
                all_func_ins[root + '/' + file] = new_function_call

    model = asm2vec.model.Asm2Vec(d=64)
    training_repo = model.make_function_repo(train_functions)
    model.train(training_repo)
    print('Training complete.')

    current_pack_file = dict()
    com_postfix = 'comment.record'
    train_function_list, train_funcs_vec = [], []
    val_function_list, val_funcs_vec = [], []
    test_function_list, test_funcs_vec = [], []
    for file in test_functions:
        print('Now with file:', file)
        pack_name = file[:file.find('_')]
        pack_name = pack_name[pack_name.rfind('/') + 1:]
        if pack_name not in current_pack_file:
            current_pack_file[pack_name] = 1
        else:
            current_pack_file[pack_name] = current_pack_file[pack_name] + 1

        if current_pack_file[pack_name] <= int(pack_file_num[pack_name] * 0.8):
            current_function_list = train_function_list
            current_funcs_vec = train_funcs_vec
        elif current_pack_file[pack_name] <= int(pack_file_num[pack_name] * 0.9):
            current_function_list = val_function_list
            current_funcs_vec = val_funcs_vec
        else:
            current_function_list = test_function_list
            current_funcs_vec = test_funcs_vec

        # Handle the function instructions
        func_ins = all_func_ins[file]
        for func in test_functions[file]:
            if func.name() not in func_ins:
                continue

            # Get addr-block dict
            addr_blocks = dict()
            current_addr = '0x00'
            for ins in func_ins[func.name()]:
                if ins.endswith(':'):
                    current_addr = ins.strip(':')
                    addr_blocks[current_addr] = []
                elif ins.startswith('      '):
                    if current_addr not in addr_blocks:
                        addr_blocks[current_addr] = []
                    temp_ins = addr_blocks[current_addr]
                    temp_ins.append(ins.strip(' '))
                    addr_blocks[current_addr] = temp_ins

            FBH_Block = []
            if FSE_FLAG:
                if len(addr_blocks) < MAX_DATAFLOW_BLOCK:
                    binary_path, _ = file.rsplit('.', 1)
                    FBH_Block = use_depgraph.depgraph_analysis(binary_path, func.name())

            vf_ins = model.to_vec_ins(func, addr_blocks, FBH_Block)
            vf_block = model.to_vec_block(func, addr_blocks, FBH_Block)
            if vf_ins.shape == vf_block.shape:
                vf_final = vf_ins + vf_block
                current_funcs_vec.append(vf_final)
                current_function_list.append(func)

    for mode in ['train', 'val', 'test']:
        if mode == 'train':
            current_function_list = train_function_list
            current_funcs_vec = train_funcs_vec
        elif mode == 'val':
            current_function_list = val_function_list
            current_funcs_vec = val_funcs_vec
        else:
            current_function_list = test_function_list
            current_funcs_vec = test_funcs_vec

        com_record = open(data_path + mode + '_comment.record', mode='w')
        emb_record = open(data_path + mode + '_embeddings.record', mode='w')
        test_sample_com = open(data_path + 'sample_comment.record', mode='a')
        test_sample_emb = open(data_path + 'sample_embeddings.record', mode='a')
        COUNT = 0
        for (ef, efv) in zip(current_function_list, current_funcs_vec):
            this_com = find_related_comment(funcname_comment, ef.name())
            if this_com != '' and this_com != '[]':
                com_record.write('%%Function: => ' + this_com + '\n')
                json.dump(efv.tolist(), emb_record)
                emb_record.write('\n')
                if COUNT < 10:
                    test_sample_com.write('%%Function: => ' + this_com + '\n')
                    json.dump(efv.tolist(), test_sample_emb)
                    test_sample_emb.write('\n')
                COUNT += 1
        com_record.close()
        emb_record.close()
        test_sample_com.close()
        test_sample_emb.close()




