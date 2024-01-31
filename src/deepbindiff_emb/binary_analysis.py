import os
import re
import ast
import json
import time
import r2pipe
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Register list
reg8_list = ['rax', 'rcx', 'rdx', 'rbx', 'rsi', 'rdi', 'rsp', 'rbp',
             'r8', 'r9', 'r10', 'r11', 'r12', 'r13', 'r14', 'r15']
reg4_list = ['eax', 'ecx', 'edx', 'ebx', 'esi', 'edi', 'esp', 'ebp',
             'r8d', 'r9d', 'r10d', 'r11d', 'r12d', 'r13d', 'r14d', 'r15d']
reg2_list = ['ax', 'cx', 'dx', 'bx', 'si', 'di', 'sp', 'bp',
             'r8w', 'r9w', 'r10w', 'r11w', 'r12w', 'r13w', 'r14w', 'r15w']
reg1_list = ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'spl', 'bpl', 'r8b',
             'r9b', 'r10b', 'r11b', 'r12b', 'r13b', 'r14b', 'r15b']
ptr_list = ['byte', 'seyte', 'word', 'sword', 'dword', 'sdword', 'fword', 'qword', 'tbyte', 'ptr']

MAXIMUM_NODE_NUM = 300
Global_Comment_count = 0

binary_path = '/binaries/'
info_path = '/data/'

def node_deduplicate(node_list):
    new_list = []
    for n in node_list:
        assert type(n) == Node
        FLAG = False
        for candidate in new_list:
            if type(candidate) == Node and candidate.Name == n.Name:
                FLAG = True
        if not FLAG:
            new_list.append(n)
    return new_list


def node_pair_deduplicate(pair_list):
    new_list = []
    for (s, e) in pair_list:
        assert type(s) == Node
        assert type(e) == Node
        if s.Start != e.Start:
            new_list.append((s, e))
    return new_list


class Node:
    def __init__(self, start_addr, input_Name=None, input_Tokens=None, input_Ins_addr=None, input_Opcode=None):
        self.Name = input_Name
        self.Tokens = input_Tokens
        self.Ins_addr = input_Ins_addr
        self.Opcodes = input_Opcode
        if input_Name is None:
            self.Name = ""
        if input_Ins_addr is None:
            self.Ins_addr = []
        if input_Tokens is None:
            self.Tokens = []
        if input_Opcode is None:
            self.Opcodes = []
        self.Start = start_addr
        self.Link_list = []

    def add_Name(self, input_Name):
        if type(input_Name) != str:
            print('ERROR IN add_Name')
        self.Name = input_Name

    def add_Tokens(self, input_Tokens):
        self.Tokens = self.Tokens + input_Tokens

    def add_Ins_addr(self, input_Ins_addr):
        if type(input_Ins_addr) != str:
            print('ERROR IN add_Ins_addr')
        self.Ins_addr.append(input_Ins_addr)

    def add_Opcode(self, input_Opcode):
        if type(input_Opcode) != str:
            print('ERROR IN add_Opcode')
        self.Opcodes.append(input_Opcode)

    def add_Link_list(self, input_list):
        if type(input_list) != list:
            print('ERROR IN add_Link_list')
        self.Link_list = node_deduplicate(self.Link_list + input_list)


def parse_instruction(ins):
    ins = re.sub('\s+', ', ', ins, 1)
    operand = ins.split(', ')
    symbols = []
    for i in range(len(operand)):
        temp_symbols = re.split('([0-9A-Za-z]+)', operand[i])
        for sym in temp_symbols:
            if sym != '':
                symbols.append(sym)
    return symbols


def find_reg(token):
    for sym in parse_instruction(token):
        sym = sym.lower()
        if sym in reg1_list:
            return 'reg1'
        elif sym in reg2_list:
            return 'reg2'
        elif sym in reg4_list:
            return 'reg4'
        elif sym in reg8_list:
            return 'reg8'
        elif sym in ptr_list:
            return ''

    return token


def normalization(token, function_var, flag, last_token='None'):
    token = token.replace(' ', '').replace(',', '')
    token = token.lower()
    ADDR_FLAG = False
    if '[' in token and ']' in token:
        ADDR_FLAG = True

    for var in function_var:
        if var in token:
            token = token.replace(var, function_var[var])

    if not flag:
        return token

    token = find_reg(token)
    if '0x' in token or '.' in token or last_token == 'call' or token.replace('-', '').isdigit():
        token = 'imme'
    elif '[' in token and ']' in token:
        token = 'var'

    if ADDR_FLAG and '[' in token: print('WARNING:', token)
    if ADDR_FLAG and '[' not in token: token = '[' + token + ']'
    return token


def SplitTokens(ins, function_var, all_nodes=None):
    tokens = []
    Bracket_count = 0
    index = 0
    last_token = 'None'
    Call_normalization_flag = True
    Comma = 0
    for i in range(len(ins)):
        if Bracket_count == 0 and ins[i] == ',':
            Comma += 1
            if Comma == 2: break
        if ins[i] == '[' or ins[i] == '(' or ins[i] == '{':
            Bracket_count = Bracket_count + 1
        elif ins[i] == ']' or ins[i] == ')' or ins[i] == '}':
            Bracket_count = Bracket_count - 1
        if ins[i] == ' ' and Bracket_count == 0:
            last_token = normalization(ins[index: i], function_var, Call_normalization_flag, last_token)
            if all_nodes is not None and last_token == 'call':
                Call_normalization_flag = False
            tokens.append(last_token)
            index = i + 1
    if index < len(ins):
        last_token = normalization(ins[index:], function_var, Call_normalization_flag, last_token)
        tokens.append(last_token)
    while '' in tokens: tokens.remove('')
    while None in tokens: tokens.remove(None)

    link_list = []
    if tokens[0] == 'call' and all_nodes is not None:
        current_name = tokens[-1].replace('[', '').replace(']', '')
        for node in all_nodes:
            if node.Name == current_name:
                link_list.append(node)

    return tokens, link_list


def OutputCount(input_opcodes_list):
    statics = {}
    for opcode in input_opcodes_list:
        if opcode not in statics:
            statics.update({opcode: 1})
        else:
            count = statics[opcode] + 1
            statics.update({opcode: count})
    return str(statics)


def CreateNodelist(pack_path, all_nodes=None):
    temp_nodelist = []  # 'list' <'str'>
    for line in open(pack_path + '.nodelist', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if '%%Function: ' not in line and line != '':
            start, _ = line.split(' ')
            temp_nodelist.append(start)

    nodelist = []  # 'list' <Node>
    function_var_dict = {}
    current_node = None
    READ_FLAG = False
    Function_name = ''
    for line in open(pack_path + '.instructions', mode='r'):
        ins = line.replace('\r', '').replace('\n', '')
        pre_index = ins.find('0x')

        index = 44
        if len(ins) >= index:
            while ins[index - 1] != ' ':
                index = index - 1

        if ins.startswith('/ '):
            # Start of the FUNCTION
            function_var_dict.clear()
            pre_i = ins.find(': ') + 2
            post_i = ins.find(' (')
            Function_name = ins[pre_i: post_i]
            Function_name = Function_name.replace(' ', '').replace(',', '')
            Function_name = Function_name.lower()

        elif ('; arg' in ins or '; var' in ins) and ' @ ' in ins:
            # Variables definition e.g., arg int64_t arg1 @ rdi
            left, right = ins.split(' @ ')
            name = left[left.rfind(' ') + 1:]
            function_var_dict.update({name: right})

        elif pre_index == 12 and '      ' in ins[pre_index:] and len(ins) > index:
            # Normal instructions
            addr = ins[pre_index:ins.find(' ', pre_index)]
            if not READ_FLAG:
                for start_addr in temp_nodelist:
                    if addr == start_addr:
                        current_node = Node(addr)
                        READ_FLAG = True
                        current_node.add_Name(Function_name)
            if READ_FLAG:
                hex_index = ins.find('0x')
                current_ins_addr = ins[hex_index: ins.find(' ', hex_index)]

                post_index = ins.find(';', index)
                if post_index == -1:
                    individual_ins = ins[index:]
                else:
                    individual_ins = ins[index: post_index]
                temp_tokens, link_list = SplitTokens(individual_ins, function_var_dict, all_nodes)
                current_node.add_Link_list(link_list)
                current_node.add_Ins_addr(current_ins_addr)
                current_node.add_Opcode(temp_tokens[0])
                current_node.add_Tokens(temp_tokens)

        elif ins == '':
            # Find the end of a Basic block
            READ_FLAG = False
            if current_node is not None:
                nodelist.append(current_node)
            current_node = None

    return nodelist


def Comment2Block(file_path):
    global Global_Comment_count
    # Read the results from the Radare2
    nodelist = CreateNodelist(file_path)
    # If "handle direct call" (e.g., call imme -> call xxx xxx xxx), uncomment this line
    # nodelist = CreateNodelist(file_path, nodelist)

    old_FunctionToNode = {}  # Dict{'str' Function_Name: 'list' <Node> Block_range}
    Func_name = 'None'
    for line in open(file_path + '.nodelist', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if '%%Function: ' in line:
            _, Func_name = line.split('%%Function: ')
            old_FunctionToNode.update({Func_name: []})
        elif line != '':
            start, end = line.split(' ')
            temp_list = old_FunctionToNode[Func_name]
            for node in nodelist:
                if node.Start == start:
                    temp_list.append(node)
            old_FunctionToNode.update({Func_name: temp_list})

    FunctionToNode = {}  # Dict{'str' Function_Name: 'list' <Node> Block_range}
    Func_name = 'None'
    for line in open(file_path + '.nodelist', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if '%%Function: ' in line:
            _, Func_name = line.split('%%Function: ')
            FunctionToNode.update({Func_name: []})
        elif line != '':
            start, end = line.split(' ')
            temp_list = FunctionToNode[Func_name]
            for node in nodelist:
                if node.Start == start:
                    temp_list.append(node)
                    for more_node in node.Link_list:
                        for ele in old_FunctionToNode:
                            if old_FunctionToNode[ele] and old_FunctionToNode[ele][0].Start == more_node.Start:
                                temp_list = temp_list + old_FunctionToNode[ele]
            FunctionToNode.update({Func_name: temp_list})

    # Collect the Edges
    old_FunctionToEdge = {}  # Dict{<str> Function_Name: <list[tuple(Node, Node)]> Edge}
    Func_name = 'None'
    for line in open(file_path + '.edgelist', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if '%%Function: ' in line:
            _, Func_name = line.split('%%Function: ')
            old_FunctionToEdge.update({Func_name: []})
        elif line != '':
            start, end = line.split(' -> ')
            start_num = int(start, 16)
            end_num = int(end, 16)
            start_node, end_node = None, None
            for node in nodelist:
                if node.Start == start:
                    start_node = node
                if node.Start == end:
                    end_node = node
            if not old_FunctionToEdge[Func_name]:
                temp_list = []
            else:
                temp_list = old_FunctionToEdge[Func_name]
            if start_node is None or end_node is None:
                continue
            temp_list.append((start_node, end_node))
            old_FunctionToEdge.update({Func_name: temp_list})

    FunctionToEdge = {}  # Dict{<str> Function_Name: <list[tuple(Node, Node)]> Edge}
    Func_name = 'None'
    added_node = []
    added_pair = []
    for line in open(file_path + '.edgelist', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if '%%Function: ' in line:
            _, Func_name = line.split('%%Function: ')
            added_node.clear()
            added_pair.clear()
            FunctionToEdge.update({Func_name: []})
        elif line != '':
            start, end = line.split(' -> ')
            start_num = int(start, 16)
            end_num = int(end, 16)
            start_node, end_node = None, None
            for node in nodelist:
                if node.Start == start:
                    start_node = node
                if node.Start == end:
                    end_node = node
            if not FunctionToEdge[Func_name]:
                temp_list = []
            else:
                temp_list = FunctionToEdge[Func_name]
            if start_node is None or end_node is None:
                continue

            Start_add_Flag = False
            that_node = None
            if start_node.Link_list:
                for more_node in start_node.Link_list:
                    for ele in old_FunctionToEdge:
                        if not old_FunctionToEdge[ele]:
                            continue
                        (s, e) = old_FunctionToEdge[ele][0]
                        if more_node.Start == s.Start:
                            if (start_node.Start, more_node.Start) not in added_pair:
                                temp_list.append((start_node, more_node))
                                added_pair.append((start_node.Start, more_node.Start))
                            if more_node.Start not in added_node:
                                temp_list = temp_list + old_FunctionToEdge[ele]
                                added_node.append(more_node.Start)
                            Start_add_Flag = True
                            (ss, ee) = old_FunctionToEdge[ele][-1]
                            that_node = ee

            if start_node.Start != end_node.Start:
                if end_node.Link_list:
                    for more_node in end_node.Link_list:
                        for ele in old_FunctionToEdge:
                            if not old_FunctionToEdge[ele]:
                                continue
                            (s, e) = old_FunctionToEdge[ele][0]
                            if more_node.Start == s.Start:
                                if (end_node.Start, more_node.Start) not in added_pair:
                                    temp_list.append((end_node, more_node))
                                    added_pair.append((end_node.Start, more_node.Start))
                                if more_node.Start not in added_node:
                                    temp_list = temp_list + old_FunctionToEdge[ele]
                                    added_node.append(more_node.Start)

                if Start_add_Flag:
                    temp_list.append((that_node, end_node))
                else:
                    temp_list.append((start_node, end_node))

            new_temp_list = node_pair_deduplicate(temp_list)
            FunctionToEdge.update({Func_name: new_temp_list})

    with open(file_path + '.snippet', mode='r') as s_file, open(file_path + '.snippet.log', mode='w') as logs:
        CommentToBlocks = {}
        CommentToEdges = {}
        for line in s_file:
            line = line.replace('\r', '').replace('\n', '')
            if '%%Function: ' in line:
                contents = line.split(' ')
                func_name = contents[1]
                hex_addr = contents[-1]
                blocks = []
                edges = []
                for ele in FunctionToNode:
                    if FunctionToNode[ele] and FunctionToNode[ele][0].Start == hex_addr:
                        blocks = FunctionToNode[ele]
                        edges = FunctionToEdge[ele]
                CommentToBlocks.update({func_name: blocks})
                CommentToEdges.update({func_name: edges})


        comment_count = 0
        for c2b, c2e in zip(CommentToBlocks, CommentToEdges):
            comment_count = comment_count + 1
            logs.write('-------------------------------------------------------------------------------------' + '\n')
            logs.write('Function: ' + c2b + '\n')
            for item in CommentToBlocks[c2b]:
                logs.write(item.Start + ': ' + ', '.join(item.Tokens) + '\n')
            logs.write('Edges: ' + '\n')
            edges = []
            for (s, e) in CommentToEdges[c2b]:
                edges.append(s.Start + ' -> ' + e.Start)
            logs.write(', '.join(edges) + '\n')
        Global_Comment_count = Global_Comment_count + comment_count

    return CommentToBlocks, CommentToEdges


def UseR2pipe(root, pack, source_file, CFG_file=info_path):
    folder = root.replace(binary_path, CFG_file) + pack + '_' + source_file.replace('.exe', '') + '/'
    if not os.path.exists(folder):
        os.makedirs(folder)

    src_path = root + pack + '/' + source_file
    dst_path = folder + pack + '_' + source_file
    print('Copy the ' + src_path + ' to the ' + dst_path)
    shutil.copyfile(src=src_path, dst=dst_path)
    dst_file_path = dst_path.replace('.exe', '')

    # Maybe can add the breakpoint here
    with open(dst_file_path + '.nodelist', mode='w') as node_file, open(dst_file_path + '.edgelist', mode='w') as e_file, \
            open(dst_file_path + '.instructions', mode='w') as i_file, open(dst_file_path + '.function', mode='w') as f_file, \
            open(dst_file_path + '.snippet', mode='w') as s_file, open(dst_file_path + '.log', mode='w') as log_file:

        r2 = r2pipe.open(dst_path)
        r2.quit()
        r2 = r2pipe.open(dst_path)
        r2.cmd('aaa')

        # Get all the functions
        r2.cmd('afl')
        function_list = []
        function_list = r2.cmd('afl').split('\n')
        function_addr_name = {}
        node_list = []

        for fun in function_list:
            fun = fun.strip('\r')
            if fun == '': continue
            addr = fun[:fun.find(' ')]
            name = fun[fun.rfind(' ') + 1:]
            f_file.write('%%Function: ' + name + ' ' + addr + ' ' + str(int(addr, 16)) + '\n')
            # if 'sym.' in name:
            temp_name = name.replace('sym.', '')
            s_file.write('%%Function: ' + temp_name + ' ' + str(int(addr, 16)) + ' ' + addr + '\n')
            function_addr_name.update({addr: name})

        for fun_addr in function_addr_name:
            node_file.write('%%Function: ' + function_addr_name[fun_addr] + '\n')
            e_file.write('%%Function: ' + function_addr_name[fun_addr] + '\n')

            r2.cmd('afb @ ' + fun_addr)
            nodes = r2.cmd('afb @ ' + fun_addr).split('\n')
            for n in nodes:
                log_file.write(n + '\n')
            CONTINUE_FLAG = False
            for n in nodes:
                n = n.replace('\r', '')
                if n == '': continue
                node_info = n.split(' ')

                # Avoid the nested functions
                if node_info[0] in function_addr_name and node_info[0] != fun_addr:
                    CONTINUE_FLAG = True
                elif node_info[0] in function_addr_name and node_info[0] == fun_addr:
                    CONTINUE_FLAG = False
                if CONTINUE_FLAG:
                    continue

                start = int(node_info[0], 16)
                end = int(node_info[1], 16)
                if start not in node_list:
                    node_list.append(start)
                out_start = hex(start)
                out_end = hex(end)
                while len(out_start) < 10:
                    out_start = '0x' + '0' + out_start.replace('0x', '')
                while len(out_end) < 10:
                    out_end = '0x' + '0' + out_end.replace('0x', '')
                node_file.write(out_start + ' ' + out_end + '\n')

                # Handle the edges
                if ' j ' in n and ' f ' not in n and ' s ' not in n:
                    next_node_1 = n[n.find(' j ') + 3:]
                    out_end_1 = hex(int(next_node_1, 16))
                    while len(out_end_1) < 10:
                        out_end_1 = '0x' + '0' + out_end_1.replace('0x', '')
                    e_file.write(out_start + ' -> ' + out_end + '\n')
                elif ' j ' in n and ' f ' in n and ' s ' not in n:
                    next_node_1 = n[n.find(' j ') + 3: n.find(' f ')]
                    next_node_2 = n[n.find(' f ') + 3:]
                    out_end_1 = hex(int(next_node_1, 16))
                    while len(out_end_1) < 10:
                        out_end_1 = '0x' + '0' + out_end_1.replace('0x', '')
                    out_end_2 = hex(int(next_node_2, 16))
                    while len(out_end_2) < 10:
                        out_end_2 = '0x' + '0' + out_end_2.replace('0x', '')
                    e_file.write(out_start + ' -> ' + out_end_1 + '\n')
                    e_file.write(out_start + ' -> ' + out_end_2 + '\n')
                elif ' j ' in n and ' f ' not in n and ' s ' in n and n.count(' s ') < 2:
                    next_node_1 = n[n.find(' j ') + 3: n.find(' s ')]
                    next_node_2 = n[n.find(' s ') + 3:]
                    out_end_1 = hex(int(next_node_1, 16))
                    while len(out_end_1) < 10:
                        out_end_1 = '0x' + '0' + out_end_1.replace('0x', '')
                    out_end_2 = hex(int(next_node_2, 16))
                    while len(out_end_2) < 10:
                        out_end_2 = '0x' + '0' + out_end_2.replace('0x', '')
                    e_file.write(out_start + ' -> ' + out_end_1 + '\n')
                    e_file.write(out_start + ' -> ' + out_end_2 + '\n')
                elif ' j ' in n and ' f ' not in n and ' s ' in n and n.count(' s ') >= 2:
                    next_node_1 = n[n.find(' j ') + 3: n.find(' s ')]
                    next_node_2 = n[n.find(' s ') + 3:].split(' s ')
                    out_end_1 = hex(int(next_node_1, 16))
                    while len(out_end_1) < 10:
                        out_end_1 = '0x' + '0' + out_end_1.replace('0x', '')
                    e_file.write(out_start + ' -> ' + out_end_1 + '\n')
                    for n_node in next_node_2:
                        out_end = hex(int(n_node, 16))
                        while len(out_end) < 10:
                            out_end = '0x' + '0' + out_end.replace('0x', '')
                        e_file.write(out_start + ' -> ' + out_end + '\n')
                elif ' s ' in n:
                    next_nodes = n.split(' s ')
                    for n_node in next_nodes[1:]:
                        out_end = hex(int(n_node, 16))
                        while len(out_end) < 10:
                            out_end = '0x' + '0' + out_end.replace('0x', '')
                        e_file.write(out_start + ' -> ' + out_end + '\n')
                else:
                    e_file.write(out_start + ' -> ' + out_start + '\n')

        Addr_comment = dict()
        for node in node_list:
            r2.cmd('pdb @ ' + hex(node))
            assembly_ins = r2.cmd('pdb @ ' + hex(node)).split('\n')
            for ins in assembly_ins:
                ins = ins.strip('\r')
                if 'str.Start_Comment_' in ins:
                    index = ins.find('str.Start_Comment_')
                    addr_s = ins[ins.find('0x'): ins.find(' ', ins.find('0x'))]
                    pre_index = ins.find(':', index)
                    comment = ins[pre_index + 1: ins.find(';', pre_index)]
                    Addr_comment[addr_s] = comment
                i_file.write(ins + '\n')

        for addr_s in Addr_comment:
            s_file.write('%%Comment: ' + Addr_comment[addr_s].replace(' ', '_') + ' %%Address: ' + addr_s + '\n')

        r2.quit()
        node_file.close()
        e_file.close()
        i_file.close()
        f_file.close()
        s_file.close()
        log_file.close()


def main():
    global Global_Comment_count

    CFG_dict = info_path

    for root, dirs, files in os.walk('../..' + binary_path):
        if not root.endswith('/'): root = root + '/'
        if root == '../..' + binary_path:
            continue
        pack = root[root[:-1].rfind('/') + 1: -1]
        for file in files:
            # if 'sg3_utils' not in root or 'sg_vpd_lib.exe' not in file: continue
            UseR2pipe('../..' + binary_path, pack, file, CFG_file=CFG_dict)
            pass

    CFG_dict = '../..' + CFG_dict

    # return -1

    for root, dirs, files in os.walk(CFG_dict):
        if not root.endswith('/'): root = root + '/'
        root = root.replace('\\\\', '/')
        if root == CFG_dict: continue

        file_name = root[root[:-1].rfind('/') + 1: -1]
        print("Handle File:", file_name)

        CommentToBlocks, CommentToEdges = Comment2Block(root + file_name)

        with open(root + 'blockIdxToTokens', mode='w') as blockIdxToTokens, \
                open(root + 'blockIdxToOpcodeNum', mode='w') as blockIdxToOpcodeNum, \
                open(root + 'blockIdxToOpcodeCounts', mode='w') as blockIdxToOpcodeCounts, \
                open(root + 'insToBlockCounts', mode='w') as insToBlockCounts, \
                open(root + 'opcodeList', mode='w') as opcodelist, \
                open(root + 'edgeList', mode='w') as edgelist, \
                open(root + 'blockToIdx', mode='w') as blockToIdx:
            for com in CommentToBlocks:
                if len(CommentToBlocks[com]) > MAXIMUM_NODE_NUM:
                    continue

                source_type = '%%Function: '
                blockIdxToTokens.write(source_type + com + '\n')
                blockIdxToOpcodeNum.write(source_type + com + '\n')
                blockIdxToOpcodeCounts.write(source_type + com + '\n')
                insToBlockCounts.write(source_type + com + '\n')
                opcodelist.write(source_type + com + '\n')
                blockToIdx.write(source_type + com + '\n')
                block_count = 0
                all_opcodes = []
                for ele in CommentToBlocks[com]:
                    blockIdxToTokens.write(str(block_count) + ': ' + ', '.join(ele.Tokens) + '\n')
                    blockIdxToOpcodeNum.write(str(block_count) + ': ' + str(len(ele.Opcodes)) + '\n')
                    blockIdxToOpcodeCounts.write(str(block_count) + ': ' + OutputCount(ele.Opcodes) + '\n')
                    blockToIdx.write(ele.Start + ' -> ' + str(block_count) + '\n')
                    opcodelist.write(', '.join(ele.Opcodes) + '\n')
                    all_opcodes = all_opcodes + ele.Opcodes
                    block_count = block_count + 1
                opcode_static = ast.literal_eval(OutputCount(all_opcodes))
                for opcode in opcode_static:
                    insToBlockCounts.write(opcode + ': ' + str(opcode_static[opcode]) + '\n')

            for com in CommentToEdges:
                if len(CommentToBlocks[com]) > MAXIMUM_NODE_NUM:
                    continue
                edgelist.write(source_type + com + '\n')
                for (s, e) in CommentToEdges[com]:
                    edgelist.write(hex(int(s.Start, 16)) + ' ' + hex(int(e.Start, 16)) + '\n')

if __name__ == '__main__':
    main()
