from __future__ import annotations
from typing import Optional

import time

from huffman import HuffmanTree
from utils import *


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([65, 66, 67, 66]))
    >>> d == {65: 1, 66: 2, 67: 1}
    True
    """
    freq_dic = {}
    for each in text:
        if each not in freq_dic:  # first time appears
            freq_dic[each] = 1
        else:
            freq_dic[each] += 1  # appeared before
    return freq_dic


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {2: 6, 3: 4}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t == result
    True
    >>> freq = {2: 6, 3: 4, 7: 5}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(2), \
                             HuffmanTree(None, HuffmanTree(3), HuffmanTree(7)))
    >>> t == result
    True
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    if len(freq_dict) == 1:  # handle the case which there is only one symbol
        symbol = list(freq_dict.keys())[0]
        d_sym = (symbol + 1) % 256  # avoid repeated symbol
        return HuffmanTree(None, HuffmanTree(symbol), HuffmanTree(d_sym))
    # the following handles cases which freq_dict has more than one symbols
    cache_huff = []  # a parallel list corresponding to cache_freq
    cache_freq = []  # a parallel list corresponding to cache_huff
    for sym in freq_dict:  # create HuffmanTree for all symbols first
        cache_huff.append(HuffmanTree(sym))
        cache_freq.append(freq_dict[sym])
    sorted_huff = cache_huff
    sorted_freq = cache_freq
    while len(sorted_huff) > 1:  # stop if there is only one tree in sorted_huff
        sorted_comp = sorted(zip(sorted_freq, sorted_huff))
        # learned zip here: https://docs.python.org/3/library/functions.html#zip
        sorted_freq, sorted_huff = zip(*sorted_comp)  # one-to-one after sorted
        sorted_freq = list(sorted_freq)
        sorted_huff = list(sorted_huff)
        f1, f2 = sorted_freq.pop(0), sorted_freq.pop(0)  # smallest 2
        lt, rt = sorted_huff.pop(0), sorted_huff.pop(0)
        new_huff = HuffmanTree(None, lt, rt)
        sorted_huff.append(new_huff)  # add new tree to the end of the list
        sorted_freq.append(f1 + f2)  # add corresponding freq to parallel list
    return sorted_huff[0]


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> d = get_codes(tree)
    >>> d == {3: "0", 2: "1"}
    True
    """
    # Cannot change the api of get_codes so use a helper function
    return __helper_get_codes(tree)


def __helper_get_codes(tree: HuffmanTree, huff_c: str = '') -> dict[int, str]:
    huff_coding = {}
    if tree.is_leaf():  # base case
        huff_coding[tree.symbol] = huff_c
        return huff_coding
    else:  # recursive case
        huff_left = __helper_get_codes(tree.left, huff_c + '0')
        huff_right = __helper_get_codes(tree.right, huff_c + '1')
        huff_coding.update(huff_left)  # use update() to combine symbols and
        huff_coding.update(huff_right)  # corresponding codes from both subtrees
        return huff_coding


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    # cannot change the api of number_nodes so use a helper function
    __helper_number_nodes(tree, 0)


def __helper_number_nodes(tree: HuffmanTree, num: int) -> Optional[int]:
    if tree.left is not None and tree.right is not None:  # base case
        if tree.left.is_leaf() and tree.right.is_leaf():
            tree.number = num
            return num + 1
        else:  # traverse in postorder for recursive case
            re_num = __helper_number_nodes(tree.left, num)
            if re_num is not None:  # in case the return value is None
                num = re_num
            re_num = __helper_number_nodes(tree.right, num)
            if re_num is not None:  # in case the return value is None
                num = re_num
            tree.number = num
            return num + 1
    else:
        return None  # PyTA complains if there is no else case


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 7, 9: 1}
    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(9)
    >>> tree = HuffmanTree(None, left, right)
    >>> avg_length(tree, freq)  # (2*2 + 7*2 + 1*1) / (2 + 7 + 1)
    1.9
    """
    huff_code_of_tree = get_codes(tree)
    weighted_sum = 0
    total_freq = 0
    for each in freq_dict:  # update weight_sum and total_freq for each symbol
        weighted_sum += freq_dict[each] * len(huff_code_of_tree[each])
        total_freq += freq_dict[each]
    return weighted_sum / total_freq


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "10", 2: "11"}
    >>> text = bytes([1, 2, 1, 0])
    >>> result = compress_bytes(text, d)
    >>> result == bytes([184])
    True
    >>> [byte_to_bits(byte) for byte in result]
    ['10111000']
    >>> text = bytes([1, 2, 1, 0, 2])
    >>> result = compress_bytes(text, d)
    >>> [byte_to_bits(byte) for byte in result]
    ['10111001', '10000000']
    """
    bits = ''.join([codes[byte] for byte in text])  # more efficient than +=
    output = []
    # extract 8 bits and convert them to 1 byte each loop:
    for index in range(0, len(bits), 8):
        output.append(bits_to_byte(bits[index: index + 8]))
    return bytes(output)


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))\
            #doctest: +NORMALIZE_WHITESPACE
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108,\
    1, 3, 1, 2, 1, 4]
    """
    if tree.number is None:  # base case
        return bytes([])
    else:  # recursive case
        output = bytes([])
        # postorder traversal
        output += tree_to_bytes(tree.left)
        output += tree_to_bytes(tree.right)
        # the following add the bytes for the root value into output
        root_bytes = []  # after traversing in postorder, handle the root
        if tree.left.is_leaf():
            root_bytes.extend([0, tree.left.symbol])
        else:
            root_bytes.extend([1, tree.left.number])
        if tree.right.is_leaf():
            root_bytes.extend([0, tree.right.symbol])
        else:
            root_bytes.extend([1, tree.right.number])
        output += bytes(root_bytes)
        return output


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 1, 1, 0)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None), \
HuffmanTree(12, None, None)), \
HuffmanTree(None, HuffmanTree(5, None, None), HuffmanTree(7, None, None)))
    """
    cache_huff = {}  # key: node number, value: HuffmanTree
    for node in range(len(node_lst)):  # convert leaves into nodes first
        if node_lst[node].l_type == 0 and node_lst[node].r_type == 0:
            cache_huff[node] = HuffmanTree(None,
                                           HuffmanTree(node_lst[node].l_data),
                                           HuffmanTree(node_lst[node].r_data))
    idx = 0  # avoid unnecessary indexing from PyTA
    for node in node_lst:  # handle internal nodes
        if node.l_type == 1 and node.r_type == 0:
            cache_huff[idx] = HuffmanTree(None,
                                          cache_huff[node.l_data],
                                          HuffmanTree(node.r_data))
        elif node.l_type == 0 and node.r_type == 1:
            cache_huff[idx] = HuffmanTree(None,
                                          HuffmanTree(node.l_data),
                                          cache_huff[node.r_data])
        elif node.l_type == 1 and node.r_type == 1:
            cache_huff[idx] = HuffmanTree(None,
                                          cache_huff[node.l_data],
                                          cache_huff[node.r_data])
        idx += 1
    return cache_huff[root_index]


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(5, None, None), \
HuffmanTree(7, None, None)), \
HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(12, None, None)))
    """
    cache = []  # store trees created, stacks not allowed so use a list
    cache.append(root_index)  # PyTA complains
    cache.pop(0)
    for RN in node_lst:
        if RN.l_type == 0 and RN.r_type == 0:  # both subtrees are leaves
            cache.append(HuffmanTree(None, HuffmanTree(RN.l_data),
                                     HuffmanTree(RN.r_data)))
        elif RN.l_type == 0 and RN.r_type == 1:  # right subtree is internal
            cache.append(HuffmanTree(None, HuffmanTree(RN.l_data),
                                     cache.pop(-1)))
        elif RN.l_type == 1 and RN.r_type == 0:  # left subtree is internal
            cache.append(HuffmanTree(None, cache.pop(-1),
                                     HuffmanTree(RN.r_data)))
        else:  # both subtrees are internal (previous 2 trees are subtrees)
            right_sub = cache.pop(-1)
            left_sub = cache.pop(-1)
            cache.append(HuffmanTree(None, left_sub, right_sub))
    return cache[0]


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.

    >>> tree = build_huffman_tree(build_frequency_dict(b'helloworld'))
    >>> number_nodes(tree)
    >>> decompress_bytes(tree, \
             compress_bytes(b'helloworld', get_codes(tree)), len(b'helloworld'))
    b'helloworld'
    """
    bits = ''.join(byte_to_bits(byte) for byte in text)  # faster than +=
    output = bytearray()  # learned bytearray object in this link:
    # https://docs.python.org/3/library/stdtypes.html#bytearray-objects
    nt = tree  # the tree we traverse for decoding
    s_out = 0  # keep track current size of output
    for bit in bits:
        if bit == '0':
            nt = nt.left
        else:
            nt = nt.right
        if nt.is_leaf():  # the base case bits decode into a byte
            output.append(nt.symbol)
            nt = tree  # look at the root node again
            s_out += 1  # update size
        if s_out == size:  # faster than len(output)
            break
    return bytes(output)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(99, None, None), \
    HuffmanTree(100, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(101, None, None), \
    HuffmanTree(None, HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 98: 23, 99: 20, 100: 16, 101: 15}
    >>> avg_length(tree, freq)
    2.49
    >>> improve_tree(tree, freq)
    >>> avg_length(tree, freq)
    2.31
    """
    sym_list = __level_order_hmt(tree)
    # sort sym_list while maintaining a one-to-one relationship:
    # source: https://docs.python.org/3/howto/sorting.html
    sorted_sym_list = [sign for frequency, sign in sorted([(freq_dict[sign],
                       sign) for sign in sym_list])]
    cur_index = -1
    # loop through sym_list and deal with each item
    while cur_index < len(sym_list) - 1:
        huff_codes = get_codes(tree)  # update coding for leaves every loop
        cur_index += 1
        sub_l, sub_sl = sym_list[:cur_index], sorted_sym_list[:cur_index]
        temp_sym = sym_list[cur_index]
        while temp_sym in sub_l:  # this key refers to some other node
            new_index = sub_l.index(temp_sym)
            temp_sym = sub_sl[new_index]
            sub_l = sym_list[new_index + 1: cur_index]
            sub_sl = sorted_sym_list[new_index + 1: cur_index]
        sorted_sym = sorted_sym_list[cur_index]
        sym_list[cur_index] = sorted_sym  # swap the keys in sym_list and sorted
        sorted_sym_list[cur_index] = temp_sym  # _sym_list
        sym_code, sorted_sym_code = huff_codes[temp_sym], huff_codes[sorted_sym]
        sym_tree = sorted_sym_tree = tree
        # the following swaps the symbol attribute
        while len(sym_code) != 0:  # set sym_tree to the node to be swapped
            if sym_code[0] == '0':
                sym_tree = sym_tree.left
                sym_code = sym_code[1:]
            else:
                sym_tree = sym_tree.right
                sym_code = sym_code[1:]
        while len(sorted_sym_code) != 0:  # set sym_tree_tree to the tree to be
            if sorted_sym_code[0] == '0':  # swapped
                sorted_sym_tree = sorted_sym_tree.left
                sorted_sym_code = sorted_sym_code[1:]
            else:
                sorted_sym_tree = sorted_sym_tree.right
                sorted_sym_code = sorted_sym_code[1:]
        sym_tree.symbol, sorted_sym_tree.symbol = sorted_sym_tree.symbol, \
            sym_tree.symbol  # swap symbols


def __level_order_hmt(tree: HuffmanTree) -> list[int]:
    """A helper function for improve tree, returns each leaf's symbol of the
    given tree in reverse order"""
    output_list = []
    cur_level = [tree]
    while len(cur_level) != 0:
        next_level = []
        for node in cur_level:
            if node.is_leaf():
                output_list.append(node.symbol)
            if not node.is_leaf() and node.right is not None:
                next_level.append(node.right)
            if not node.is_leaf() and node.left is not None:
                next_level.append(node.left)
        cur_level = next_level
    output_list.reverse()  # output should be from right to left
    return output_list


if __name__ == "__main__":
    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
