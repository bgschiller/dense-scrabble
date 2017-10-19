import string
import heapq
import sys
import marisa_trie
import random
from itertools import groupby, product
from collections import namedtuple
from nose.tools import assert_equal


class MinHeap(object):
    def __init__(self, values=(), key=None):
        self._key = key or (lambda v: None)
        self._heap = [(self._key(v), v) for v in values]
        heapq.heapify(self._heap)

    def pop(self):
        return heapq.heappop(self._heap)[1]

    def push(self, val):
        heapq.heappush(self._heap, (self._key(val), val))

    def top(self):
        if self._heap:
            return self._heap[0][1]
        else:
            return None

    def pop_value(self, val):
        """
        Do a pop, assuming the value is equal to the top.

        Otherwise, raise an error.
        """
        if self.top() == val:
            return self.pop()[1]
        self._heap.remove((self._key(val), val))
        heapq.heapify(self._heap)


WordStart = namedtuple('WordStart', 'prefix next_loc')


class Board(object):
    def __init__(self, size):
        # size by size nested list
        self.size = size
        self.rows = [[None] * size for _ in range(size)]
        self.settled = set()
        self.frontier = MinHeap(
            product(range(size), range(size)),
            # sort by distance from (0, 0)
            key=lambda pair: pair[0] ** 2 + pair[1] ** 2)

    def transpose(self):
        t = Board(0)
        t.rows = list(map(list, zip(*self.rows)))
        return t

    def words(self):
        yield from self._words()
        for word, (cix, rix) in self.transpose()._words():
            yield WordStart(word, (rix, cix))  # swap rix, cix back

    def _words(self):
        for row_ix, row in enumerate(self.rows):
            # groupby splits each row at position of None
            for key, group in groupby(
                    enumerate(row),
                    lambda pair: pair[1] is not None):
                if key:
                    group = list(group)
                    word = ''.join(char for _, char in group)
                    col_ix = max(ix for ix, _ in group)
                    yield WordStart(word, (row_ix, col_ix + 1))

    def place_letter(self, row, col, val):
        self.rows[row][col] = val

        if (row, col) not in self.settled:
            self.settled.add((row, col))
            self.frontier.pop_value((row, col))

    def unplace(self, row, col):
        self.rows[row][col] = None
        self.settled.remove((row, col))
        self.frontier.push((row, col))

    def next_position(self):
        return self.frontier.top()

    def __repr__(self):
        return '\n'.join(
            ' '.join(val or ' ' for val in row)
            for row in self.rows
        )


def test_transpose():
    b = Board(2)
    b.rows = [
        ['a', 'b'],
        ['c', None]
    ]
    assert_equal(b.transpose().rows, [
        ['a', 'c'],
        ['b', None],
    ])


def test_words():
    b = Board(4)
    b.rows = [
        ['a', 'b', None, 'c'],
        ['d', None, 'e', 'f'],
        ['g', 'h', 'i', 'j'],
        [None, 'k', 'l', None],
    ]
    assert_equal(
        set(b.words()), {
            'ab', 'c', 'd', 'ef', 'ghij', 'kl',
            'adg', 'b', 'hk', 'eil', 'cfj'
        })


SIZE = 6


def read_words():
    with open('/usr/share/dict/scrabble') as f:
        trie = marisa_trie.Trie(
            line.strip('\n') + '.'
            for line in f
            if len(line.strip('\n')) <= SIZE
        )
    return trie


LETTERS = list(string.ascii_lowercase + '.')
WORD_TRIE = read_words()


def valid_board(board, words=WORD_TRIE):
    for ws in board.words():
        could_become = words.keys(ws.prefix)
        if not could_become:
            return False
        elif (len(could_become) == 1 and
              #  minus 1 because trie has . at the end of each word.
              len(could_become[0]) - 1 > len(ws.prefix)):
            # only one possibility for rest of word -- place it!
            return finish_word(board, ws, could_become[0])

    return True


def finish_word(board, ws, word):
    next_letter = word[len(ws.prefix)]
    board.place_letter(*ws.next_loc, next_letter)
    if assume_letter(board):
        # we did it!
        return True
    else:
        # whoops, that must not have worked out...
        board.unplace(*ws.next_loc)


def assume_letter(board):
    coords = board.next_position()
    if not coords:
        # no work to be done. woo hoo, we did it!
        return True
    random.shuffle(LETTERS)
    for letter in LETTERS:
        board.place_letter(*coords, letter)
        if valid_board(board) and assume_letter(board):
            # woo hoo, we did it!
            return True
        # rats, if we haven't returned by this point then
        # we made a mistake. Either our board wasn't valid,
        # or assume_letter() eventually ran into difficulty.
        # Oh well, we're about to try a different letter on
        # the next iteration anyways...
    # if we get to *this* point, then dang...
    # not a single one of the letters we tried produced a valid
    # board. Better let our caller know so they can deal with the
    # situation. But first, let's unplace our letter.
    board.unplace(*coords)
    if coords[0] == coords[1]:
        print(board, file=sys.stderr)
        print(file=sys.stderr)
    return False


def main():
    b = Board(SIZE)
    valid = assume_letter(b)
    if not valid:
        print('unable to make a board...')
    else:
        print(b)


if __name__ == '__main__':
    main()
