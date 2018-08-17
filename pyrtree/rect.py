import math


class Rect(object):
    """
    A rectangle class that stores: an axis aligned rectangle, and: two
     flags (swapped_x and swapped_y).  (The flags are stored
     implicitly via swaps in the order of minx/y and maxx/y.)
    """

    __slots__ = ("x", "y", "xx", "yy", "swapped_x", "swapped_y")

    def __init__(self, minx, miny, maxx, maxy):
        self.swapped_x = (maxx < minx)
        self.swapped_y = (maxy < miny)
        self.x = minx
        self.y = miny
        self.xx = maxx
        self.yy = maxy

        if self.swapped_x:
            self.x, self.xx = maxx, minx
        if self.swapped_y:
            self.y, self.yy = maxy, miny

    def __repr__(self):
        return '<Rect({}, {}, {}, {})>'.format(self.x, self.y, self.xx, self.yy)

    def coords(self):
        return self.x, self.y, self.xx, self.yy

    def overlap(self, orect):
        return self.intersect(orect).area()

    def write_raw_coords(self, toarray, idx):
        if self.swapped_x:
            # 叶子节点
            toarray[idx] = self.xx
            toarray[idx + 2] = self.x
        else:
            toarray[idx] = self.x
            toarray[idx + 2] = self.xx

        if self.swapped_y:
            toarray[idx + 1] = self.yy
            toarray[idx + 3] = self.y
        else:
            toarray[idx + 1] = self.y
            toarray[idx + 3] = self.yy

    def area(self):
        """
        矩形面积
        :return:
        """
        w = self.xx - self.x
        h = self.yy - self.y
        return w * h

    def extent(self):
        x = self.x
        y = self.y
        return x, y, self.xx - x, self.yy - y

    def grow(self, amt):
        a = amt * 0.5
        return Rect(self.x - a, self.y - a, self.xx + a, self.yy + a)

    def intersect(self, o):
        """
        获取两个矩形相交的部分(矩形)
        :param o:
        :return:
        """
        if self is NullRect:
            return NullRect
        if o is NullRect:
            return NullRect

        nx, ny = max(self.x, o.x), max(self.y, o.y)
        nx2, ny2 = min(self.xx, o.xx), min(self.yy, o.yy)
        w, h = nx2 - nx, ny2 - ny

        if w <= 0 or h <= 0:
            return NullRect

        return Rect(nx, ny, nx2, ny2)

    def does_contain(self, o):
        return self.does_containpoint((o.x, o.y)) and self.does_containpoint((o.xx, o.yy))

    def does_intersect(self, o):
        return self.intersect(o).area() > 0

    def does_containpoint(self, p):
        x, y = p
        return (x >= self.x and x <= self.xx and y >= self.y and y <= self.yy)

    def union(self, o):
        if o is NullRect:
            return Rect(self.x, self.y, self.xx, self.yy)
        if self is NullRect:
            return Rect(o.x, o.y, o.xx, o.yy)

        x = self.x
        y = self.y
        xx = self.xx
        yy = self.yy
        ox = o.x
        oy = o.y
        oxx = o.xx
        oyy = o.yy

        nx = x if x < ox else ox
        ny = y if y < oy else oy
        nx2 = xx if xx > oxx else oxx
        ny2 = yy if yy > oyy else oyy

        res = Rect(nx, ny, nx2, ny2)

        return res

    def union_point(self, o):
        x, y = o
        return self.union(Rect(x, y, x, y))

    def diagonal_sq(self):
        if self is NullRect: return 0
        w = self.xx - self.x
        h = self.yy - self.y
        return w * w + h * h

    def diagonal(self):
        return math.sqrt(self.diagonal_sq())


NullRect = Rect(0.0, 0.0, 0.0, 0.0)
NullRect.swapped_x = False
NullRect.swapped_y = False


def union_all(kids):
    cur = NullRect
    for k in kids:
        cur = cur.union(k.rect)
    assert not cur.swapped_x
    return cur


if __name__ == '__main__':
    a = Rect(0, 0, 5, 5)
    b = Rect(1, 1, 3, 3)
    c = a.intersect(b)
    print(c)

