import os
import sys
import csv
import argparse
import psutil
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds


def get_platform():
    if sys.platform.startswith('win'):
        return 'windows'
    elif sys.platform.startswith('linux'):
        return 'linux'
    else:
        return 'none'


def compile_code(source_path):
    platform = get_platform()
    name = source_path[:source_path.rfind('.')]
    if platform == 'windows':
        res = os.system('cl.exe "' + source_path + '" /openmp')
        if res != 0:
            print('error while compiling, please check the output for error messages')
            executable = ''
        else:
            os.remove(name + '.obj')
            executable = name + '.exe'
    elif platform == 'linux':
        res = os.system('gcc -Wall -fopenmp -m64 -O3 ' + source_path + ' -o ' + name + ' -lm -lpthread')
        if res != 0:
            print('error while compiling ' + source_path + ', please check the output for error messages')
            executable = ''
        else:
            executable = name
    else:
        executable = ''
    return executable


def get_executable(program_name):
    platform = get_platform()
    if platform == 'windows':
        executable = program_name + '.exe'
    else:
        executable = './' + program_name

    if os.path.exists(executable) is False:
        if os.path.exists('./' + program_name + '.c'):
            executable = compile_code(('' if platform == 'windows' else './') + program_name + '.c')
        else:
            executable = ''
    return executable


def run_vocab(executable_path, input_path, output_path, min_count, max_vocab, verbose):
    if executable_path == '' or os.path.exists(executable_path) is False:
        print('neither vocab_count executable nor its source code exist, please copy vocab_count.c beside this python file')
        sys.exit()

    command = executable_path
    command += ' -input "' + input_path + '"'
    command += ' -output "' + output_path + '"'
    if min_count > 0:
        command += ' -min-count ' + str(min_count)
    if max_vocab > 0:
        command += ' -max-vocab ' + str(max_vocab)
    if verbose >= 0:
        command += ' -verbose ' + str(verbose)

    if verbose > 0:
        print(command)
    if os.path.exists(output_path) is True:
        print(output_path + ' exists and will be replaced.')
    res = os.system(command)
    if res != 0:
        print('an error occurred, exiting.')
        sys.exit()


def run_cooccur(executable_path, input_path, vocab_path, output_path, window_size, dist_weight, verbose):
    if executable_path == '' or os.path.exists(executable_path) is False:
        print('neither cooccurrence executable nor its source code exist, please copy cooccurrence.c beside this python file')
        sys.exit()
    memory_available = min((psutil.virtual_memory().available * 0.8 / (1024 * 1024 * 1024), 256))

    command = executable_path + ' -vocab-file "' + vocab_path + '"'
    command += ' -input "' + input_path + '"'
    command += ' -output "' + output_path + '"'
    if window_size > 0:
        command += ' -window-size ' + str(window_size)
    if dist_weight >= 0:
        command += ' -distance-weighting ' + str(dist_weight)
    if verbose >= 0:
        command += ' -verbose ' + str(verbose)
    command += ' -memory ' + str(memory_available)
    command += ' -overflow-file overflow_' + str(np.random.randint(1000000, 9999999, 1)[0])

    if verbose > 0:
        print(command)
    if os.path.exists(output_path) is True:
        print(output_path + ' exists and will be replaced.')
    res = os.system(command)
    if res != 0:
        print('an error occurred, exiting.')
        sys.exit()


def run_svdns_c(executable_path, input_path, vocab_path, output_path, pmi_cutoff, pmi_shift, dimension, thread, verbose):
    if executable_path == '' or os.path.exists(executable_path) is False:
        print('neither svdns executable nor its source code exist, please copy svdns.c beside this python file')
        sys.exit()

    command = executable_path
    command += ' -input "' + input_path + '"'
    command += ' -vocab "' + vocab_path + '"'
    command += ' -output "' + output_path + '"'
    command += ' -pmicutoff ' + str(pmi_cutoff)
    command += ' -shift ' + str(pmi_shift)
    if dimension > 0:
        command += ' -dimension ' + str(dimension)
    if thread > 1:
        command += ' -thread ' + str(thread)
    if verbose >= 0:
        command += ' -verbose ' + ('1' if verbose > 0 else '0')

    if verbose > 0:
        print(command)
    if os.path.exists(output_path) is True:
        print(output_path + ' exists and will be replaced.')
    res = os.system(command)
    if res != 0:
        print('an error occurred, exiting.')
        sys.exit()


def calc_pmi(data, pmi_cutoff=None, pmi_shift=None):
    sum_row = np.squeeze(np.asarray(np.sum(data, axis=1)))
    sum_sum = np.sum(sum_row)
    prob_row = sum_row / sum_sum
    prob_col = np.squeeze(np.asarray(np.sum(data, axis=0))) / sum_sum
    pmi = np.log2(data.data / (sum_sum * prob_row[data.row] * prob_col[data.col]))

    if pmi_cutoff is not None:
        idx_good = (pmi >= pmi_cutoff)
        pmi = pmi[idx_good]
        idx_rowcol = (data.row[idx_good], data.col[idx_good])
    else:
        idx_rowcol = (data.row, data.col)

    if pmi_shift is not None:
        pmi += pmi_shift

    return coo_matrix((pmi, idx_rowcol), data.shape)


def load_binary_cooccurrence(file_path):
    dt = np.dtype([('row', 'int32'), ('col', 'int32'), ('val', 'float64')])
    data = np.fromfile(file_path, dtype=dt)
    shape = (int(np.max(data['row'])), int(np.max(data['col'])))
    data = coo_matrix((data['val'], (data['row'] - 1, data['col'] - 1)), shape=shape)
    return data


def run_svdns_python(input_path, vocab_path, output_path, pmi_cutoff, pmi_shift, dimension, verbose):
    try:
        char_encoding = 'utf-8'
        df_vocab = pd.read_csv(vocab_path, header=None, dtype=str, sep=' ', quoting=csv.QUOTE_NONE, quotechar='', encoding=char_encoding)
    except:
        char_encoding = 'cp1252'
        df_vocab = pd.read_csv(vocab_path, header=None, dtype=str, sep=' ', quoting=csv.QUOTE_NONE, quotechar='', encoding=char_encoding)

    data = load_binary_cooccurrence(input_path)
    cnt_cooccur = data.nnz
    if verbose is not None and verbose > 0:
        print('Calculating Pointwise Mutual Information (PMI) ...')
        print(str(cnt_cooccur) + ' records in the cooccurrence file.')
    data = calc_pmi(data, pmi_cutoff, pmi_shift)
    if verbose is not None and verbose > 0:
        print(str(data.nnz) + ' records in the PMI matrix.')
        print('Generating word embeddings ...')

    u, _, _ = svds(data.tocsr(), dimension)
    if verbose is not None and verbose > 0:
        print('Writing the embeddings to file ...')
    pd.DataFrame(np.hstack((df_vocab.values[:, 0].reshape(-1, 1), u))).to_csv(output_path, header=None, index=False, float_format='%8.7f', sep=' ', quoting=csv.QUOTE_NONE, quotechar='', encoding=char_encoding)
    if verbose is not None and verbose > 0:
        print('All done!')


def main():
    exec_vocab = get_executable('vocab')
    exec_cooccur = get_executable('cooccurrence')
    exec_svdns = get_executable('svdns')

    parser = argparse.ArgumentParser(description='SVD-NS program for learning word embeddings.')
    parser.add_argument('-input', required=False, dest='input', default='corpus.txt')
    parser.add_argument('-vocab', required=False, dest='vocab', default='')
    parser.add_argument('-cooccur', required=False, dest='cooccur', default='')
    parser.add_argument('-output', required=False, dest='output', default='embedding.txt')
    parser.add_argument('-maxvocab', required=False, dest='max_vocab', type=int, default=0)
    parser.add_argument('-mincount', required=False, dest='min_count', type=int, default=10)
    parser.add_argument('-distweight', required=False, dest='dist_weight', type=int, default=1)
    parser.add_argument('-windowsize', required=False, dest='window_size', type=int, default=10)
    parser.add_argument('-pmicutoff', required=False, dest='pmi_cutoff', type=float, default=-2.5)
    parser.add_argument('-pmishift', required=False, dest='pmi_shift', type=float, default=2.5)
    parser.add_argument('-dimension', required=False, dest='dimension', type=int, default=100)
    parser.add_argument('-engine', required=False, dest='engine', type=str, default='auto')
    parser.add_argument('-thread', required=False, dest='thread', type=int, default=1)
    parser.add_argument('-verbose', required=False, dest='verbose', type=int, default=2)
    args = parser.parse_args(sys.argv[1:])

    if os.path.exists(args.input):
        if args.vocab == '' or os.path.exists(args.vocab) is False:
            args.vocab = 'vocab.txt' if args.vocab == '' else args.vocab
            run_vocab(exec_vocab, input_path=args.input, output_path=args.vocab, min_count=args.min_count, max_vocab=args.max_vocab, verbose=args.verbose)
        run_cooccur(exec_cooccur, input_path=args.input, vocab_path=args.vocab, output_path=('cooccurrence.bin' if args.cooccur == '' else args.cooccur), window_size=args.window_size, dist_weight=args.dist_weight, verbose=args.verbose)
        if args.engine.lower() == 'python' or (args.engine.lower() == 'auto' and mp.cpu_count() > 4):
            run_svdns_python(input_path=('cooccurrence.bin' if args.cooccur == '' else args.cooccur), vocab_path=args.vocab, output_path=args.output, pmi_cutoff=args.pmi_cutoff, pmi_shift=args.pmi_shift, dimension=args.dimension, verbose=args.verbose)
        else:
            run_svdns_c(exec_svdns, input_path=('cooccurrence.bin' if args.cooccur == '' else args.cooccur), vocab_path=args.vocab, output_path=args.output, pmi_cutoff=args.pmi_cutoff, pmi_shift=args.pmi_shift, dimension=args.dimension, thread=args.thread, verbose=args.verbose)
        if args.cooccur == '':
            os.remove('cooccurrence.bin')
    elif args.cooccur != '' and os.path.exists(args.cooccur):
        if os.path.exists(args.vocab) is True:
            if args.engine.lower() == 'python' or (args.engine.lower() == 'auto' and mp.cpu_count() > 4):
                run_svdns_python(input_path=args.cooccur, vocab_path=args.vocab, output_path=args.output, pmi_cutoff=args.pmi_cutoff, pmi_shift=args.pmi_shift, dimension=args.dimension, verbose=args.verbose)
            else:
                run_svdns_c(exec_svdns, input_path=args.cooccur, vocab_path=args.vocab, output_path=args.output, pmi_cutoff=args.pmi_cutoff, pmi_shift=args.pmi_shift, dimension=args.dimension, thread=args.thread, verbose=args.verbose)
        else:
            print('when providing the cooccurrence (-cooccur) without specifying the corpus (-input), you also need to provide the vocab file using -vocab')
    else:
        print('please specify either a valid corpus using -input or a valid cooccurrence file using -cooccur')


if __name__ == "__main__":
    main()
