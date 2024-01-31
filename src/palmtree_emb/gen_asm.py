import os
import re
import random
import pickle
import math
import binaryninja
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from itertools import product
from sklearn.decomposition import PCA
from collections import Counter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

LEAST_INS = 10

# Register list
reg_list = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp',
            'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15',
            'eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp',
            'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d',
            'ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp',
            'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w',
            'al', 'cl', 'dl', 'bl', 'sil', 'dil', 'bpl', 'spl', 'r8b',
            'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']


def parse_instruction(ins, symbol_map, string_map):
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        symbols = re.split('([0-9A-Za-z]+)', operand[i])
        for j in range(len(symbols)):
            if symbols[j][:2] == '0x' and len(symbols[j]) >= 6:
                if int(symbols[j], 16) in symbol_map:
                    symbols[j] = "symbol" # function names
                elif int(symbols[j], 16) in string_map:
                    symbols[j] = "string" # constant strings
                else:
                    symbols[j] = "address" # addresses
        operand[i] = ' '.join(symbols)
    opcode = parts[0]
    return ' '.join([opcode]+operand)


def split_instruction(ins):
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    symbols = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        temp_symbols = operand[i].split(' ')
        for sym in temp_symbols:
            if sym != '':
                symbols.append(sym)
    return symbols


def cfg_random_walk(g, length, symbol_map, string_map):
    sequence = []
    for n in g:
        if n != -1 and 'text' in g._node[n]:
            s = []
            l = 0
            s.append(parse_instruction(g._node[n]['text'], symbol_map, string_map))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    if 'text' in g._node[cur]:
                        s.append(parse_instruction(g._node[cur]['text'], symbol_map, string_map))
                        l += 1
                    else:
                        break
                else:
                    break
            sequence.append(s)
        if len(sequence) > 5000:
            return sequence[:5000]
    return sequence


def dfg_random_walk(g, length, symbol_map, string_map):
    sequence = []
    for n in g:
        if n != -1 and g._node[n]['text'] != None:
            s = []
            l = 0
            s.append(parse_instruction(g._node[n]['text'], symbol_map, string_map))
            cur = n
            while l < length:
                nbs = list(g.successors(cur))
                if len(nbs):
                    cur = random.choice(nbs)
                    s.append(parse_instruction(g._node[cur]['text'], symbol_map, string_map))
                    l += 1
                else:
                    break
            sequence.append(s)
    return sequence


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


def valid_callee_expansion(caller_ins, callee_ins):
    caller_length = len(caller_ins)
    callee_length = len(callee_ins)
    if callee_length > (caller_length * 0.6):
        return False
    if caller_length < LEAST_INS:
        return False
    return True


def generate_block_arg_info_binaryninja(data_path, package_name, file_name, bin_view):
    print('Collect the block and arguments info of', file_name)
    block_folder = data_path + 'block_info/'
    arg_folder = data_path + 'arg_info/'
    if not os.path.exists(block_folder):
        os.makedirs(block_folder)
    if not os.path.exists(arg_folder):
        os.makedirs(arg_folder)

    block_info_out = open(block_folder + package_name + '_' + file_name + '.txt', 'w')
    arg_info_out = open(arg_folder + package_name + '_' + file_name + '.txt', 'w')

    for func in bin_view.functions:
        # Get block info
        block_info_out.write('%%Function: => ' + func.name + '\n')
        arg_info_out.write('%%Function: => ' + func.name + '\n')

        nodelist = []  # 'list' <'dict' <addr: ins>>
        for block in func:
            temp_node = dict()  # 'dict' <'addr: ins'>
            curr = block.start
            for inst in block:
                temp_node[hex(curr)] = bin_view.get_disassembly(curr)
                block_info_out.write(hex(curr) + ' #=># ' + bin_view.get_disassembly(curr) + '\n')
                curr += inst[1]
            block_info_out.write('##EndOfBlock##\n')
            nodelist.append(temp_node)

        # Get entry argument reference
        def find_reg(arg_name, symbols, out_file):
            for sym in symbols:
                for reg in reg_list:
                    if reg in sym.lower():
                        out_file.write(sym + ' #=># ' + arg_name + '\n')
                        return 1

        for arg in func.parameter_vars:
            arg_name = arg.var_name_and_type._source_type
            mlil_refs = func.get_mlil_var_refs(arg)
            if mlil_refs and mlil_refs[0]:
                for node in nodelist:
                    if hex(mlil_refs[0].address) in node:
                        symbols = split_instruction(node[hex(mlil_refs[0].address)])
                        symbols.reverse()
                        find_reg(arg_name, symbols, arg_info_out)

    block_info_out.close()
    arg_info_out.close()


def process_cfg_file(f, window_size, func_com, mode='train', my_func_ins=None):
    cfg_count = 0
    if mode == 'prepare':
        func_ins = dict()
    symbol_map = {}
    string_map = {}
    print('CFG:', f)
    bv = binaryninja.load(f)

    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    function_graphs = {}
    block_range = dict()  # dict() <Start_addr, End_addr>

    for func in bv.functions:
        num_block = 0
        for block in func:
            block_range[block.start] = block.end
            num_block += 1

        if num_block < 2:
            for block in func:
                if int(block.end) - int(block.start) < LEAST_INS:
                    num_block = -1
            if num_block == -1:
                continue

        G = nx.DiGraph()
        label_dict = {}
        add_map = {}
        for block in func:
            curr = block.start
            predecessor = curr
            for inst in block:
                label_dict[curr] = bv.get_disassembly(curr)
                G.add_node(curr, text=bv.get_disassembly(curr), addr=hex(curr))
                if curr != block.start:
                    G.add_edge(predecessor, curr)
                predecessor = curr
                curr += inst[1]
            for edge in block.outgoing_edges:
                G.add_edge(predecessor, edge.target.start)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    if mode == 'prepare':
        generate_block_arg_info_binaryninja(data_path, f.split('/')[-2], f.split('/')[-1], bv)
    else:
        pair_out = open(data_path + 'cfg_' + mode + '_pair.txt', 'a')
        single_out = open(data_path + 'cfg_' + mode + '_single.txt', 'a')
        single_new_out = open(data_path + 'cfg_' + mode + '_single_new.txt', 'a')
        single_new_out.write('%%BinaryFile: => ' + f + '\n')

    for name, graph in function_graphs.items():
        this_com = find_related_comment(func_com, name)

        if mode == 'prepare':
            this_com = 'Pass'

        func_addr = ''
        instructions = []

        if this_com != '' and this_com != '[]':
            if 'prepare' not in mode:
                single_new_out.write('%%Function: => ' + name + '\n')

            addr_ins = dict()
            sort_addr = []
            for n in graph:
                if n != -1 and 'text' in graph._node[n]:
                    current_addr = int(graph._node[n]['addr'], 16)
                    current_ins = graph._node[n]['text']
                    addr_ins[current_addr] = current_ins
                    sort_addr.append(current_addr)
            sort_addr.sort()
            func_addr = sort_addr[0]

            current_range = (-2, -1)
            RANGE_FLAG = False
            for addr in sort_addr:
                if addr < current_range[0] or addr >= current_range[1]:
                    RANGE_FLAG = True
                    if addr in block_range:
                        assert addr < block_range[addr]
                        current_range = (addr, block_range[addr])
                    else:
                        print('WARNING!', addr, current_range)
                        for start in block_range:
                            end = block_range[start]
                            if start <= addr < end:
                                current_range = (start, end)

                current_ins = addr_ins[addr]
                output = parse_instruction(current_ins, symbol_map, string_map)
                if 'prepare' not in mode:
                    if RANGE_FLAG:
                        single_new_out.write('%%Addr: => ' + hex(current_range[0]) + '\n')
                        RANGE_FLAG = False

                    if 'call' in current_ins:
                        elements = current_ins.split(' ')
                        try:
                            callee = int(elements[-1], 16)
                        except Exception:
                            callee = None
                        if callee is not None and my_func_ins is not None:
                            if callee in my_func_ins and func_addr in my_func_ins:
                                if valid_callee_expansion(my_func_ins[func_addr], my_func_ins[callee]):
                                    for line in my_func_ins[callee]:
                                        if '%%Addr: => ' not in line:
                                            single_out.write(line + '\n')
                                        single_new_out.write(line + '\n')
                                    single_new_out.write('%%Addr: => ' + hex(current_range[0]) + '\n')
                                    continue

                    single_out.write(output + '\n')
                    single_new_out.write(output + '\n')

                elif mode == 'prepare':
                    if RANGE_FLAG:
                        instructions.append('%%Addr: => ' + hex(current_range[0]))
                        RANGE_FLAG = False
                    instructions.append(output)
            cfg_count += 1

            if 'prepare' not in mode:
                sequence = cfg_random_walk(graph, 40, symbol_map, string_map)
                for s in sequence:
                    if len(s) >= 4:
                        for idx in range(0, len(s)):
                            for j in range(1, window_size + 1):
                                if idx - j > 0:
                                    pair_out.write(s[idx - j] + '\t' + s[idx] + '\n')
                                if idx + j < len(s):
                                    pair_out.write(s[idx] + '\t' + s[idx + j] + '\n')

        if mode == 'prepare':
            func_ins[func_addr] = instructions

    if 'prepare' not in mode:
        pair_out.close()
        single_out.close()
        single_new_out.close()

    if mode == 'prepare':
        return cfg_count, func_ins
    else:
        return cfg_count


def process_dfg_file(f, func_com, mode='train'):
    dfg_count = 0
    symbol_map = {}
    string_map = {}
    print('DFG:', f)
    bv = binaryninja.load(f)

    for sym in bv.get_symbols():
        symbol_map[sym.address] = sym.full_name
    for string in bv.get_strings():
        string_map[string.start] = string.value

    function_graphs = {}

    for func in bv.functions:
        num_block = 0
        for block in func:
            num_block += 1
        if num_block < 2:
            for block in func:
                if int(block.end) - int(block.start) < LEAST_INS:
                    num_block = -1
            if num_block == -1:
                continue

        G = nx.DiGraph()
        G.add_node(-1, text='entry_point')
        line = 0
        label_dict = {}
        label_dict[-1] = 'entry_point'

        try:
            for block in func.mlil:
                for ins in block:
                    G.add_node(ins.address, text=bv.get_disassembly(ins.address))
                    label_dict[ins.address] = bv.get_disassembly(ins.address)
                    depd = []
                    for var in ins.vars_read:
                        depd = [(func.mlil[i].address, ins.address)
                                for i in func.mlil.get_var_definitions(var)
                                if func.mlil[i].address != ins.address]
                    for var in ins.vars_written:
                        depd += [(ins.address, func.mlil[i].address)
                                 for i in func.mlil.get_var_uses(var)
                                 if func.mlil[i].address != ins.address]
                    if depd:
                        G.add_edges_from(depd)
        except Exception:
            return dfg_count

        for node in G.nodes:
            if not G.in_degree(node):
                G.add_edge(-1, node)
        if len(G.nodes) > 2:
            function_graphs[func.name] = G

    dfg_out = open(data_path + 'dfg_' + mode + '.txt', 'a')

    for name, graph in function_graphs.items():
        this_com = find_related_comment(func_com, name)
        if this_com != '' and this_com != '[]':
            dfg_count += 1

            sequence = dfg_random_walk(graph, 40, symbol_map, string_map)
            for s in sequence:
                if len(s) >= 2:
                    for idx in range(1, len(s)):
                        dfg_out.write(s[idx - 1] + '\t' + s[idx] + '\n')
    dfg_out.close()
    return dfg_count


def clear_file(data, add=''):
    cfg_train_pair = open(data + 'cfg_' + add + 'train_pair.txt', 'w')
    cfg_val_pair = open(data + 'cfg_' + add + 'val_pair.txt', 'w')
    cfg_test_pair = open(data + 'cfg_' + add + 'test_pair.txt', 'w')

    cfg_train_single = open(data + 'cfg_' + add + 'train_single.txt', 'w')
    cfg_val_single = open(data + 'cfg_' + add + 'val_single.txt', 'w')
    cfg_test_single = open(data + 'cfg_' + add + 'test_single.txt', 'w')

    cfg_train_single_new = open(data + 'cfg_' + add + 'train_single_new.txt', 'w')
    cfg_val_single_new = open(data + 'cfg_' + add + 'val_single_new.txt', 'w')
    cfg_test_single_new = open(data + 'cfg_' + add + 'test_single_new.txt', 'w')

    dfg_train = open(data + 'dfg_' + add + 'train.txt', 'w')
    dfg_val = open(data + 'dfg_' + add + 'val.txt', 'w')
    dfg_test = open(data + 'dfg_' + add + 'test.txt', 'w')

    cfg_train_pair.close()
    cfg_val_pair.close()
    cfg_test_pair.close()

    cfg_train_single.close()
    cfg_val_single.close()
    cfg_test_single.close()

    cfg_train_single_new.close()
    cfg_val_single_new.close()
    cfg_test_single_new.close()

    dfg_train.close()
    dfg_val.close()
    dfg_test.close()


if __name__ == '__main__':
    global data_path
    data_path = '../../data/'

    np.seterr(divide='ignore', invalid='ignore')
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
    parser.add_argument('--FSE', '-f', required=False, action='store_true', help='Whether enable the FSE module.')
    args = parser.parse_args()
    FSE_FLAG = args.FSE

    bin_folder = '../../binaries/'

    if FSE_FLAG:
        print('Enable the FSE mode.')
    else:
        print('Disable the FSE mode.')

    file_lst = []

    str_counter = Counter()
    window_size = 1
    for parent, dirs, files in os.walk(bin_folder):
        for f in files:
            file_lst.append(os.path.join(parent, f))

    funcname_comment = get_funcname_comment_dict(data_path)

    dfg_train = open(data_path + 'dfg_train.txt', 'w')
    dfg_train.close()

    pack_file_all_count = dict()
    file_func_ins = dict()
    i = 0
    print('Preparation...')
    for f in file_lst:
        pack_name = f[:f.rfind('/')]
        this_pack = pack_name[pack_name.rfind('/') + 1:]

        _, func_ins_dict = process_cfg_file(f, window_size, funcname_comment, 'prepare')
        file_func_ins[f] = func_ins_dict

        if this_pack not in pack_file_all_count:
            pack_file_all_count[this_pack] = 1
        else:
            pack_file_all_count[this_pack] = pack_file_all_count[this_pack] + 1
        i += 1

    clear_file(data_path)

    i = 0
    print('Generation...')
    pack_file_count = dict()
    set_count = {'train': 0, 'val': 0, 'test': 0}
    for f in file_lst:
        pack_name = f[:f.rfind('/')]
        this_pack = pack_name[pack_name.rfind('/') + 1:]

        if this_pack not in pack_file_all_count:
            print('WARNING! PACK:', this_pack)

        if this_pack not in pack_file_count:
            pack_file_count[this_pack] = 1

        if pack_file_count[this_pack] <= int(0.8 * pack_file_all_count[this_pack]):
            print(i, '/', len(file_lst), 'In the Training set')
            if FSE_FLAG:
                c_count = process_cfg_file(f, window_size, funcname_comment, 'train', file_func_ins[f])
            else:
                c_count = process_cfg_file(f, window_size, funcname_comment, 'train')
            d_count = process_dfg_file(f, funcname_comment, 'train')
            set_count['train'] = set_count['train'] + c_count
        elif pack_file_count[this_pack] <= int(0.9 * pack_file_all_count[this_pack]):
            print(i, '/', len(file_lst), 'In the Validation set')
            if FSE_FLAG:
                c_count = process_cfg_file(f, window_size, funcname_comment, 'val', file_func_ins[f])
            else:
                c_count = process_cfg_file(f, window_size, funcname_comment, 'val')
            d_count = process_dfg_file(f, funcname_comment, 'val')
            set_count['val'] = set_count['val'] + c_count
        else:
            print(i, '/', len(file_lst), 'In the Test set')
            if FSE_FLAG:
                c_count = process_cfg_file(f, window_size, funcname_comment, 'test', file_func_ins[f])
            else:
                c_count = process_cfg_file(f, window_size, funcname_comment, 'test')
            d_count = process_dfg_file(f, funcname_comment, 'test')
            set_count['test'] = set_count['test'] + c_count

        pack_file_count[this_pack] = pack_file_count[this_pack] + 1
        i += 1

    new_all_count = 0
    for my_set in set_count:
        new_all_count = new_all_count + set_count[my_set]
