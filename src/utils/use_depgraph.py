from __future__ import print_function
from builtins import range
import re
import os
import time
import signal
import threading
from pdb import pm
from binaryninja import *
import json

from future.utils import viewitems

from miasm.analysis.machine import Machine
from miasm.analysis.binary import Container
from miasm.analysis.depgraph import DependencyGraph
from miasm.expression.expression import ExprMem, ExprId, ExprInt
from miasm.core.locationdb import LocationDB

"""
parser = ArgumentParser(description="Dependency grapher")
parser.add_argument("filename", help="Binary to analyse")
parser.add_argument("func_addr", help="Function address")
parser.add_argument("target_addr", help="Address to start")
# parser.add_argument("element", nargs="+", help="Elements to track")
parser.add_argument("-m", "--architecture",
                    help="Architecture (%s)" % Machine.available_machine())
parser.add_argument("-i", "--implicit", help="Use implicit tracking",
                    action="store_true")
parser.add_argument("--unfollow-mem", help="Stop on memory statements",
                    action="store_true")
parser.add_argument("--unfollow-call", help="Stop on call statements",
                    action="store_true")
parser.add_argument("--do-not-simplify", help="Do not simplify expressions",
                    action="store_true")
parser.add_argument("--rename-args",
                    help="Rename common arguments (@32[ESP_init] -> Arg1)",
                    action="store_true")
parser.add_argument("--json",
                    help="Output solution in JSON",
                    action="store_true")
args = parser.parse_args()
"""


class MyTread(threading.Thread):
    def __init__(self, target, args=()):
        """ Self defined Threading to make it has return value"""
        super(MyTread, self).__init__()
        self.result = None
        self.func = target
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def _handle_timeout(signum, frame):
    raise TimeoutError('Timeout')


def parse_instruction(ins):
    ins = re.sub('\s+', ', ', ins, 1)
    parts = ins.split(', ')
    operand = []
    symbols = []
    if len(parts) > 1:
        operand = parts[1:]
    for i in range(len(operand)):
        temp_symbols = re.split('([0-9A-Za-z]+)', operand[i])
        for sym in temp_symbols:
            if sym != '':
                symbols.append(sym)
    return symbols


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


def related_to_var(start_sol, end_sol, my_arg, addr_ins):
    if not start_sol:
        return False
    result = []
    for this_start in start_sol:
        this_end = end_sol[0]

        changed_token = list()
        token_count = 999
        for end in end_sol:
            assert len(this_start) == len(end)
            temp_changed_token = list()
            for ele in end:
                if ele in this_start and this_start[ele] != end[ele]:
                    temp_changed_token.append(ele)
            if len(temp_changed_token) < token_count:
                token_count = len(temp_changed_token)
                changed_token = temp_changed_token
                this_end = dict()
                for ele in end:
                    this_end[ele] = end[ele].replace(' ', '')

        RESULT_FLAG = False
        if changed_token:
            for reg in my_arg:
                if RESULT_FLAG: break
                for ele in changed_token:
                    if 'CALL' in this_end[ele]:
                        continue
                    if reg.upper() in this_end[ele].upper():
                        result.append(True)
                        RESULT_FLAG = True
                        break
        else:
            symbols = split_instruction(addr_ins)
            for sym in symbols:
                if RESULT_FLAG: break
                sym = sym.upper()
                if sym in this_end:
                    for reg in my_arg:
                        if 'CALL' in this_end[sym]:
                            continue
                        if reg.upper() in this_end[sym].upper():
                            result.append(True)
                            RESULT_FLAG = True
                            break
    if False not in result and len(result) >= len(start_sol):
        return True
    return False


def obtain_block_info_binaryninja(file_path, info_path, func_name):
    nodelist = []  # 'list' <'dict' <addr: ins>>
    reg_arg_dict = {}

    block_info_path = info_path + 'block_info/' + file_path.split('/')[-2] + '_' + file_path.split('/')[-1] + '.txt'
    arg_info_path = info_path + 'arg_info/' + file_path.split('/')[-2] + '_' + file_path.split('/')[-1] + '.txt'
    if not os.path.exists(block_info_path):
        print('No find block_info_path:', block_info_path)
    if not os.path.exists(arg_info_path):
        print('No find arg_info_path:', arg_info_path)

    temp_node = dict()  # 'dict' <'addr: ins'>
    current_func = 'NoneFunction'
    for line in open(block_info_path, mode='r'):
        line = line.strip('\n')
        if '%%Function: => ' in line:
            current_func = line.replace('%%Function: => ', '')
        if current_func.lower() != func_name.lower():
            continue

        if ' #=># ' in line:
            parts = line.split(' #=># ')
            temp_node[parts[0]] = parts[-1]
        elif '##EndOfBlock##' in line:
            nodelist.append(temp_node)
            temp_node = dict()  # 'dict' <'addr: ins'>

    current_func = 'NoneFunction'
    for line in open(arg_info_path, mode='r'):
        line = line.strip('\n')
        if '%%Function: => ' in line:
            current_func = line.replace('%%Function: => ', '')
        if current_func.lower() != func_name.lower():
            continue

        if ' #=># ' in line:
            parts = line.split(' #=># ')
            reg_arg_dict[parts[0]] = parts[-1]

    return nodelist, reg_arg_dict


def obtain_block_info_radare2(file_path, func_name):
    temp_nodelist = []  # 'list' <'str'>
    for line in open(file_path + '.nodelist', mode='r'):
        line = line.replace('\r', '').replace('\n', '')
        if '%%Function: ' not in line and line != '':
            start, _ = line.split(' ')
            temp_nodelist.append(start)

    nodelist = []  # 'list' <'dict' <addr: ins>>
    temp_node = dict()  # 'dict' <'addr: ins'>
    reg_arg_dict = {}
    Function_name, start_addr, end_addr = '', '', ''
    func_name = func_name.lower()

    for line in open(file_path + '.instructions', mode='r'):
        ins = line.replace('\r', '').replace('\n', '')
        pre_index = ins.find('0x')

        index = 44
        if len(ins) >= index:
            while ins[index - 1] != ' ':
                index = index - 1

        if ins.startswith('/ '):
            pre_i = ins.find(': ') + 2
            post_i = ins.find(' (')
            Function_name = ins[pre_i: post_i]
            Function_name = Function_name.replace(' ', '').replace(',', '')
            Function_name = Function_name.lower()

        if func_name == Function_name or func_name == Function_name.replace('sym.', ''):
            if '; arg' in ins and ' @ ' in ins:
                left, right = ins.split(' @ ')
                name = left[left.rfind(' ') + 1:]
                reg_arg_dict[right] = name

            elif pre_index == 12 and '      ' in ins[pre_index:] and len(ins) > index:
                addr = ins[pre_index:ins.find(' ', pre_index)]
                post_index = ins.find(';', index)
                if post_index == -1:
                    individual_ins = ins[index:]
                else:
                    individual_ins = ins[index: post_index]
                temp_node[addr] = individual_ins

            elif ins == '':
                if temp_node:
                    nodelist.append(temp_node)
                temp_node = dict()

    return nodelist, reg_arg_dict


def depgraph_analysis(input_file, info_path, my_func_name, MAX_SOL=10):
    FSE_Block_list = []

    loc_db = LocationDB()
    if os.path.exists(input_file) and input_file.endswith('.exe'):
        with open(input_file, "rb") as fstream:
            cont = Container.from_stream(fstream, loc_db)
    else:
        with open(input_file + '.exe', "rb") as fstream:
            cont = Container.from_stream(fstream, loc_db)

    machine = Machine(cont.arch)

    elements = set()
    regs = machine.mn.regs.all_regs_ids_byname
    for ele in regs:
        if ele.islower() and ele.endswith('f'):
            continue
        elements.add(regs[ele])

    mdis = machine.dis_engine(cont.bin_stream, dont_dis_nulstart_bloc=True, loc_db=loc_db)
    lifter = machine.lifter_model_call(loc_db)

    blocks, reg_var = obtain_block_info_binaryninja(input_file, info_path, my_func_name)
    if not blocks:
        return FSE_Block_list
    elif not reg_var:
        # No arguments
        for addrs in blocks:
            FSE_Block_list.append(list(addrs.keys())[0])
        return FSE_Block_list

    if blocks and blocks[0] and list(blocks[0].keys())[0]:
        asmcfg = mdis.dis_multiblock(int(str(list(blocks[0].keys())[0]), 0))
    else:
        # No arguments
        for addrs in blocks:
            FSE_Block_list.append(list(addrs.keys())[0])
        return FSE_Block_list

    try:
        ircfg = lifter.new_ircfg_from_asmcfg(asmcfg)
    except:
        for addrs in blocks:
            FSE_Block_list.append(list(addrs.keys())[0])
        return FSE_Block_list

    dg = DependencyGraph(
        ircfg, implicit=False,
        apply_simp=True,
        follow_mem=False,
        follow_call=False
    )

    for addrs in blocks:
        if not addrs: continue
        var_sol = list()  # list <sol>
        previous_addr = ''
        for addr in addrs:
            target_addr = int(str(addr), 0)

            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(1)

            try:
                current_loc_key = next(iter(ircfg.getby_offset(target_addr)))
                assignblk_index = 0
                current_block = ircfg.get_block(current_loc_key)
                for assignblk_index, assignblk in enumerate(current_block):
                    if assignblk.instr.offset == target_addr:
                        break

                all_result = dg.get(current_block.loc_key, elements, assignblk_index, set())
                temp_var_sol = list()  # list <dict -> reg: var>
                for sol_nb, sol in enumerate(all_result):
                    if sol_nb >= MAX_SOL:
                        break
                    results = sol.emul(lifter, ctx={})
                    temp_var_sol.append({str(k).upper(): str(v).upper() for k, v in viewitems(results)})

                current_var_sol = temp_var_sol
                if current_var_sol is None or not current_var_sol:
                    raise Exception('Timeout')

            except Exception:
                FSE_Block_list.append(list(addrs.keys())[0])
                break
            finally:
                signal.alarm(0)

            if var_sol and current_var_sol and previous_addr:
                if related_to_var(var_sol, current_var_sol, reg_var, addrs[previous_addr]):
                    FSE_Block_list.append(list(addrs.keys())[0])
                    break
            var_sol = current_var_sol
            previous_addr = addr

    return FSE_Block_list
