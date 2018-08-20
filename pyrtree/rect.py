import math


class Rectangle(object):
    __slots__ = ("x", "y", "xx", "yy", "swapped_x", "swapped_y")

    def __init__(self, minx, miny, maxx, maxy):
        """
        minx,miny为矩形左下角的点，maxx, maxy为矩形右上角的点
        :param minx:
        :param miny:
        :param maxx:
        :param maxy:
        """
        self.swapped_x = maxx < minx
        self.swapped_y = maxy < miny
        if self.swapped_x:
            self.x, self.xx = maxx, minx
        else:
            self.x, self.xx = minx, maxx

        if self.swapped_y:
            self.y, self.yy = maxy, miny
        else:
            self.y, self.yy = miny, maxy

    def __repr__(self):
        return '<Rectangle({}, {}, {}, {})>'.format(self.x, self.y, self.xx, self.yy)

    def coords(self):
        return self.x, self.y, self.xx, self.yy

    def overlap(self, target_rect):
        """
        重叠面积
        :param target_rect:
        :return:
        """
        return self.intersect(target_rect).area()

    def overlap_rate(self, target_rect):
        """
        重叠率
        :param target_rect:
        :return:
        """
        return self.overlap(target_rect) / min(self.area(), target_rect.area())

    def write_raw_coords(self, rect_array, idx):
        if self.swapped_x:
            rect_array[idx] = self.xx
            rect_array[idx + 2] = self.x
        else:
            rect_array[idx] = self.x
            rect_array[idx + 2] = self.xx

        if self.swapped_y:
            rect_array[idx + 1] = self.yy
            rect_array[idx + 3] = self.y
        else:
            rect_array[idx + 1] = self.y
            rect_array[idx + 3] = self.yy

    def area(self):
        """
        矩形面积
        :return:
        """
        w = self.xx - self.x
        h = self.yy - self.y
        return w * h

    def extent(self):
        """
        返回左下角坐标和矩形的宽和高
        :return:
        """
        x = self.x
        y = self.y
        return x, y, self.xx - x, self.yy - y

    def intersect(self, o):
        """
        获取两个矩形相交部分(矩形)
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

        return Rectangle(nx, ny, nx2, ny2)

    def does_contain(self, object_rect):
        """
        判断当前矩形是否整个包含目标矩形
        :param object_rect:
        :return:
        """
        return self.does_contain_point((object_rect.x, object_rect.y)) and self.does_contain_point((object_rect.xx, object_rect.yy))

    def does_intersect(self, object_rect):
        """
        判断当前矩形是否和目标矩形有重叠
        :param object_rect:
        :return:
        """
        return self.intersect(object_rect).area() > 0

    def does_contain_point(self, p):
        """
        判断当前矩形是否包含某一点
        :param p:
        :return:
        """
        x, y = p
        return self.x <= x <= self.xx and self.y <= y <= self.yy

    def union(self, object_rect):
        """
        获取两个矩形的并集，会生成一个包括两个矩形的新矩形
        :param object_rect:
        :return:
        """
        if object_rect is NullRect:
            return Rectangle(self.x, self.y, self.xx, self.yy)
        if self is NullRect:
            return Rectangle(object_rect.x, object_rect.y, object_rect.xx, object_rect.yy)

        x = self.x
        y = self.y
        xx = self.xx
        yy = self.yy
        ox = object_rect.x
        oy = object_rect.y
        oxx = object_rect.xx
        oyy = object_rect.yy

        nx = x if x < ox else ox
        ny = y if y < oy else oy
        nx2 = xx if xx > oxx else oxx
        ny2 = yy if yy > oyy else oyy

        res = Rectangle(nx, ny, nx2, ny2)

        return res

    def diagonal_sq(self):
        """
        返回矩形对角线长度的平方
        :return:
        """
        if self is NullRect:
            return 0
        w = self.xx - self.x
        h = self.yy - self.y
        return w * w + h * h

    def diagonal(self):
        """
        返回矩形对角线长度
        :return:
        """
        return math.sqrt(self.diagonal_sq())


NullRect = Rectangle(0.0, 0.0, 0.0, 0.0)
NullRect.swapped_x = False
NullRect.swapped_y = False


def union_all(kids):
    cur = NullRect
    for k in kids:
        cur = cur.union(k.rect)
    assert not cur.swapped_x
    return cur


if __name__ == '__main__':
    a = Rectangle(0, 0, 5, 5)
    b = Rectangle(1, 1, 3, 3)
    c = a.intersect(b)
    print(c)

