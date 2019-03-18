from random import randint
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

img_size = 42
max_holes = 10
min_width = 6
EDGE = {0: 'down', 1: 'up', 2: 'right', 3: 'left'}


def gen_square():
    # make the base square
    sq = np.zeros([img_size, img_size], dtype=int)
    x1 = randint(1, 3 * img_size // 4)
    y1 = randint(1, 3 * img_size // 4)
    x2 = randint(x1 + min_width, img_size - 2)
    y2 = randint(y1 + min_width, img_size - 2)
    for i in range(x1, x2 + 1, 1):
        sq[i][y1] = sq[i][y2] = 1
    for j in range(y1, y2 + 1, 1):
        sq[x1][j] = sq[x2][j] = 1

    n_holes = randint(1, max_holes + 1)
    holes = []
    for h in range(n_holes):
        e_h = EDGE[randint(0, 3)]

        if e_h == 'down':
            prev = [s for s in holes if 'down' in s]
            if len(prev) == 2:
                continue
            if len(prev) == 1:
                if prev[0][1] - x1 > 5:
                    w1 = randint(x1 + 2, prev[0][1] - 4)
                    w2 = randint(w1 + 2, prev[0][1] - 2)
                elif x2 - prev[0][2] > 5:
                    w1 = randint(prev[0][2] + 2, x2 - 4)
                    w2 = randint(w1 + 2, x2 - 2)
                else:
                    continue
            else:
                w1 = randint(x1 + 2, x2 - 4)
                w2 = randint(w1 + 2, x2 - 2)
            l = randint(1, y2 - y1 - 1)
            stop = l + y1
            for j in range(y1, y1 + l + 1):
                if stop != l + y1:
                    break
                sq[w1][j] = sq[w2][j] = 1
                for i in range(w1, w2 + 1):
                    if sq[i][j + 2] == 1 or sq[w1 - 1][j + 1] == 1 or sq[w2 + 1][j + 1] == 1:
                        stop = j

            for i in range(w1, w2 + 1):
                sq[i][y1] = 0
                sq[i][stop] = 1
                sq[w1][y1] = sq[w2][y1] = 1

            holes.append(['down', w1, w2, stop])

        if e_h == 'up':
            prev = [s for s in holes if 'up' in s]
            if len(prev) == 2:
                continue
            if len(prev) == 1:
                if prev[0][1] - x1 > 5:
                    w1 = randint(x1 + 2, prev[0][1] - 4)
                    w2 = randint(w1 + 2, prev[0][1] - 2)
                elif x2 - prev[0][2] > 5:
                    w1 = randint(prev[0][2] + 2, x2 - 4)
                    w2 = randint(w1 + 2, x2 - 2)
                else:
                    continue
            else:
                w1 = randint(x1 + 2, x2 - 4)
                w2 = randint(w1 + 2, x2 - 2)
            l = randint(1, y2 - y1 - 1)
            stop = y2 - l
            for j in range(y2, y2 - l - 1, -1):
                if stop != y2 - l:
                    break
                sq[w1][j] = sq[w2][j] = 1
                for i in range(w1, w2 + 1):
                    if sq[i][j - 2] == 1 or sq[w1 - 1][j - 1] == 1 or sq[w2 + 1][j - 1] == 1:
                        stop = j

            for i in range(w1, w2 + 1):
                sq[i][y2] = 0
                sq[i][stop] = 1
                sq[w1][y2] = sq[w2][y2] = 1

            holes.append(['up', w1, w2, stop])

        if e_h == 'right':
            prev = [s for s in holes if 'right' in s]
            if len(prev) == 2:
                continue
            if len(prev) == 1:
                if prev[0][1] - y1 > 5:
                    w1 = randint(y1 + 2, prev[0][1] - 4)
                    w2 = randint(w1 + 2, prev[0][1] - 2)
                elif y2 - prev[0][2] > 5:
                    w1 = randint(prev[0][2] + 2, y2 - 4)
                    w2 = randint(w1 + 2, y2 - 2)
                else:
                    continue
            else:
                w1 = randint(y1 + 2, y2 - 4)
                w2 = randint(w1 + 2, y2 - 2)
            l = randint(1, x2 - x1 - 1)
            stop = x2 - l
            for j in range(x2, x2 - l - 1, -1):
                if stop != x2 - l:
                    break
                sq[j][w1] = sq[j][w2] = 1
                for i in range(w1, w2 + 1):
                    if sq[j - 2][i] == 1 or sq[j - 1][w1 - 1] == 1 or sq[j - 1][w2 + 1] == 1:
                        stop = j

            for i in range(w1, w2 + 1):
                sq[x2][i] = 0
                sq[stop][i] = 1
                sq[x2][w1] = sq[x2][w2] = 1

            holes.append(['right', w1, w2, stop])

        if e_h == 'left':
            prev = [s for s in holes if 'left' in s]
            if len(prev) == 2:
                continue
            if len(prev) == 1:
                if prev[0][1] - y1 > 5:
                    w1 = randint(y1 + 2, prev[0][1] - 4)
                    w2 = randint(w1 + 2, prev[0][1] - 2)
                elif y2 - prev[0][2] > 5:
                    w1 = randint(prev[0][2] + 2, y2 - 4)
                    w2 = randint(w1 + 2, y2 - 2)
                else:
                    continue
            else:
                w1 = randint(y1 + 2, y2 - 4)
                w2 = randint(w1 + 2, y2 - 2)
            l = randint(1, x2 - x1 - 1)
            stop = x1 + l
            for j in range(x1, x1 + l + 1):
                if stop != x1 + l:
                    break
                sq[j][w1] = sq[j][w2] = 1
                for i in range(w1, w2 + 1):
                    if sq[j + 2][i] == 1 or sq[j + 1][w1 - 1] == 1 or sq[j + 1][w2 + 1] == 1:
                        stop = j

            for i in range(w1, w2 + 1):
                sq[x1][i] = 0
                sq[stop][i] = 1
                sq[x1][w1] = sq[x1][w2] = 1

            holes.append(['left', w1, w2, stop])

    return sq




print(gen_square())

