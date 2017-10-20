import string
import sys
import marisa_trie
import random
from itertools import groupby, product
from collections import namedtuple
from nose.tools import assert_equal

WordStart = namedtuple('WordStart', 'prefix next_loc step')


class BoardError(ValueError):
    pass


def dist(tup):
    return (tup[0] ** 2 + tup[1] ** 2)


class Board(object):
    def __init__(self, size):
        # size by size nested list
        self.size = size
        self.rows = [[None] * size for _ in range(size)]
        self.settled = set()
        self.frontier = set(product(range(size), range(size)))

    def transpose(self):
        t = Board(0)
        t.rows = list(map(list, zip(*self.rows)))
        return t

    def words(self):
        yield from self._words()
        for word, (cix, rix), (dx, dy) in self.transpose()._words():
            yield WordStart(word, (rix, cix), (dy, dx))  # swap rix, cix back

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
                    yield WordStart(word, (row_ix, col_ix + 1), (0, 1))

    def place_letter(self, row, col, val):
        if self.rows[row][col]:
            raise BoardError('that position already has a different letter')
        self.rows[row][col] = val
        self.settled.add((row, col))
        self.frontier.remove((row, col))

    def sanity_check(self):
        assert len(self.settled | self.frontier) == self.size * self.size
        assert self.settled & self.frontier == set()
        for row_ix, row in enumerate(self.rows):
            for col_ix, val in enumerate(row):
                if val is not None:
                    assert (
                        row_ix == 0 or col_ix == 0 or
                        self.rows[row_ix - 1][col_ix] is not None or
                        self.rows[row_ix][col_ix - 1] is not None)

    def unplace(self, row, col):
        """
        clear everything below and right of (row, col)
        """
        for row_ix in range(row, SIZE):
            for col_ix in range(col, SIZE):
                self.rows[row_ix][col_ix] = None
                self.settled.discard((row_ix, col_ix))
                self.frontier.add((row_ix, col_ix))

    def next_position(self):
        try:
            return min(self.frontier, key=dist)
        except ValueError:
            return None

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
    board.sanity_check()
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
    next_letters = word[len(ws.prefix):-1]
    row_ix, col_ix = ws.next_loc
    drow, dcol = ws.step
    to_unplace = []
    try:
        for ix, letter in enumerate(next_letters):
            loc = (row_ix + drow * ix, col_ix + dcol * ix)
            if loc[0] >= SIZE or loc[1] >= SIZE:
                # This word wants to continue off the size of
                # the board
                return False
            board.place_letter(*loc, letter)
            to_unplace.append(loc)
        board.sanity_check()
    except BoardError:
        return False
    if valid_board(board) and assume_letter(board):
        # we did it!
        return True
    else:
        # whoops, that must not have worked out...
        board.sanity_check()
        for loc in to_unplace:
            board.unplace(*loc)
        board.sanity_check()


def assume_letter(board):
    coords = board.next_position()
    if not coords:
        # no work to be done. woo hoo, we did it!
        return True
    random.shuffle(LETTERS)
    for letter in LETTERS:
        board.place_letter(*coords, letter)
        board.sanity_check()
        if valid_board(board) and assume_letter(board):
            # woo hoo, we did it!
            return True
        # rats, if we haven't returned by this point then
        # we made a mistake. Either our board wasn't valid,
        # or assume_letter() eventually ran into difficulty.
        # Oh well, we're about to try a different letter on
        # the next iteration anyways...
        board.unplace(*coords)
        board.sanity_check()
    # if we get to *this* point, then dang...
    # not a single one of the letters we tried produced a valid
    # board. Better let our caller know so they can deal with the
    # situation.
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
