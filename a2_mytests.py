from __future__ import annotations
import pytest
from compress import *


def test_build_frequency_dict():
    """test build_frequency_dict works on a larger set of data"""
    test_list = bytes([33, 72, 198, 198, 33, 72, 72, 107, 33, 198, 107, 72, 33])
    dictionary = build_frequency_dict(test_list)
    result = {33: 4, 72: 4, 198: 3, 107: 2}
    assert dictionary == result


def test_build_huffman_tree():
    """test if build_huffman_tree works for tree with left branch as symbol
    and the right branch as another HMT"""
    freq = {9: 2, 20: 6, 69: 15, 78: 13, 134: 9, 117: 10}
    t = build_huffman_tree(freq)
    result = HuffmanTree(None, HuffmanTree(None, HuffmanTree(117, None, None),
                                           HuffmanTree(78, None, None)),
                         HuffmanTree(None, HuffmanTree(69, None, None),
                                     HuffmanTree(None, HuffmanTree(None,
                                                                   HuffmanTree(
                                                                       9,
                                                                       None,
                                                                       None),
                                                                   HuffmanTree(
                                                                       20, None,
                                                                       None)),
                                                 HuffmanTree(134, None, None))))
    assert t == result


def test_get_codes():
    """test get-codes on a larger set of data"""
    freq = {9: 2, 20: 6, 69: 15, 78: 13, 134: 9, 117: 10}
    t = build_huffman_tree(freq)
    p = get_codes(t)
    result = {117: '00', 78: '01', 69: '10', 134: '111', 9: '1100', 20: '1101'}
    assert p == result


def test_number_nodes():
    """test number_nodes works for
    if trees with left as symbol and right as internal node"""
    t = HuffmanTree(None, HuffmanTree(7), HuffmanTree(None, HuffmanTree(None,
                    HuffmanTree(3), HuffmanTree(2)), HuffmanTree(None,
                    HuffmanTree(9), HuffmanTree(10))))
    number_nodes(t)
    assert t.number == 3
    assert t.left.number is None
    assert t.right.number == 2
    assert t.right.right.number == 1
    assert t.right.left.number == 0


def test_avg_length():
    """test if avg_length works on a large data set"""
    freq = {9: 2, 20: 6, 69: 15, 78: 13, 134: 9, 117: 10}
    t = build_huffman_tree(freq)
    al = avg_length(t, freq)
    result = (10 * 2 + 13 * 2 + 15 * 2 + 9 * 3 + 2 * 4 + 6 * 4) / (
            10 + 13 + 15 + 9 + 2 + 6)
    assert al == result


def test_compress_bytes():
    """test if compress_bytes works on some other numbers"""
    d = {0: '00', 1: '01', 2: '10', 3: '1100', 4: '111', 5: '1101'}
    text1 = bytes([5])
    result1 = compress_bytes(text1, d)
    text2 = bytes([3, 4])
    result2 = compress_bytes(text2, d)
    text3 = bytes([0, 3, 4, 1, 2, 5])
    result3 = compress_bytes(text3, d)
    assert [byte_to_bits(r1) for r1 in result1] == ['11010000']
    assert [byte_to_bits(r2) for r2 in result2] == ['11001110']
    assert [byte_to_bits(r3) for r3 in result3] == ['00110011', '10110110',
                                                    '10000000']


def test_tree_to_bytes():
    """test if tree_to_bytes works on a tree with left branch as symbol
    and right branch as another HMT"""
    tree_2 = HuffmanTree(None, HuffmanTree(148), HuffmanTree(None,
                                                             HuffmanTree(107),
                                                             HuffmanTree(223)))
    number_nodes(tree_2)
    assert list(tree_to_bytes(tree_2)) == [0, 107, 0, 223, 0, 148, 1, 0]


def test_generate_tree_general():
    """test generate_tree_general on a larger input"""
    read_nodes = [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114),
                  ReadNode(1, 0, 1, 1), ReadNode(0, 100, 0, 111),
                  ReadNode(0, 108, 1, 3), ReadNode(1, 2, 1, 4)]
    tree_g = generate_tree_general(read_nodes, 5)
    tree_correct = HuffmanTree(None, HuffmanTree(None, HuffmanTree(None,
                               HuffmanTree(104), HuffmanTree(101)),
                               HuffmanTree(None, HuffmanTree(119),
                               HuffmanTree(114))),HuffmanTree(None,
                               HuffmanTree(108), HuffmanTree(None,
                               HuffmanTree(100), HuffmanTree(111))))
    assert tree_g == tree_correct


def test_generate_tree_postorder():
    """test generate_tree_postorder on a larger input"""
    tree_test = HuffmanTree(None, HuffmanTree(None, HuffmanTree(None,
                            HuffmanTree(104), HuffmanTree(101)),
                            HuffmanTree(None, HuffmanTree(119),
                            HuffmanTree(114))), HuffmanTree(None,
                            HuffmanTree(108), HuffmanTree(None,
                            HuffmanTree(100), HuffmanTree(111))))
    lst_test = [ReadNode(0, 104, 0, 101), ReadNode(0, 119, 0, 114),
                ReadNode(1, 0, 1, 1), ReadNode(0, 100, 0, 111),
                ReadNode(0, 108, 1, 3), ReadNode(1, 2, 1, 4)]
    assert generate_tree_postorder(lst_test, 5) == tree_test


def test_decompress_bytes():
    """test if decompress bytes works for all kind of inputs"""
    tree_1b = build_huffman_tree(build_frequency_dict(b'k'))
    number_nodes(tree_1b)
    output_1b = decompress_bytes(tree_1b, compress_bytes(b'k',
                                 get_codes(tree_1b)), len(b'k'))
    tree_2b = build_huffman_tree(build_frequency_dict(b'KA-GA-YA-KI'))
    number_nodes(tree_2b)
    output_2b = decompress_bytes(tree_2b, compress_bytes(b'KA-GA-YA-KI',
                                 get_codes(tree_2b)), len(b'KA-GA-YA-KI'))
    tree_3b = build_huffman_tree(build_frequency_dict(b'I am dian gun'))
    output_3b = decompress_bytes(tree_3b, compress_bytes(b'I am dian gun',
                                 get_codes(tree_3b)), len(b'I am dian gun'))
    assert output_1b == b'k'
    assert output_2b == b'KA-GA-YA-KI'
    assert output_3b == b'I am dian gun'


def test_improve_tree():
    """test if improve_tree works on a more complicated tree"""
    fd = {9: 2, 20: 6, 134: 9, 117: 10, 78: 13, 69: 15}
    t_improved = build_huffman_tree(fd)
    tree_for_test = HuffmanTree(None, HuffmanTree(None, HuffmanTree(9),
                                HuffmanTree(134)), HuffmanTree(None,
                                HuffmanTree(20), HuffmanTree(None,
                                HuffmanTree(None, HuffmanTree(69),
                                HuffmanTree(117)), HuffmanTree(78))))
    improve_tree(tree_for_test, fd)
    assert tree_for_test == t_improved


if __name__ == '__main__':
    pytest.main(['a2_my_tests.py'])
