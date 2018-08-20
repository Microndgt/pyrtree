## R-tree.
# see doc/ref/r-tree-clustering-split-algo.pdf
import random
import array
from .rect import Rectangle, union_all, NullRect

MAX_CHILDREN = 10
MAX_KMEANS = 5


class RTree(object):
    def __init__(self):
        self.count = 0
        self.leaf_count = 0
        self.rect_pool = array.array('d')  # double
        self.node_pool = array.array('L')  # long
        self.leaf_pool = []
        self.cursor = _NodeCursor.create(self, NullRect)

    def _ensure_pool(self, idx):
        """
        分配pool
        :param idx:
        :return:
        """
        if len(self.rect_pool) < (4 * idx):
            self.rect_pool.extend([0, 0, 0, 0] * idx)
            self.node_pool.extend([0, 0] * idx)

    def insert(self, obj, obj_rect):
        self.cursor.insert(obj, obj_rect)
        assert self.cursor.index == 0

    def query_rect(self, rect):
        for x in self.cursor.query_rect(rect):
            yield x

    def query_point(self, point):
        for x in self.cursor.query_point(point):
            yield x

    def walk(self, predicate):
        return self.cursor.walk(predicate)


class _NodeCursor(object):
    """
    _NodeCursor对象是对RTree中某个节点的代理，用于retrieve相关节点的数据
    _NodeCursor对象共享RTree的pool数据，因此对_NodeCursor对象的save就等同于RTree pool数据的save
    """
    __slots__ = ("root", "npool", "rpool", "index", "rect", "next_sibling", "first_child")

    @classmethod
    def create(cls, rooto, rect):
        """
        在rooto上创建一个节点，如果需要，扩展rooto的pool(Node pool, rect pool)大小以容纳新的节点
        创建新的_NodeCursor对象，其root为rooto节点，idx为当前rooto的第几个元素，rect为传过来的rect数据
        first_child, next_sibling为默认值0，相关pool数据继承rooto的数据
        最后将新创建的节点数据写入到创建的叶子节点的数据中
        :param rooto:
        :param rect:
        :return:
        """
        # idx在RTree中是唯一存在的，因此分配新的节点后会依次加入到pool中
        # 通过索引first_child, next_sibling来建立节点的相互关系
        idx = rooto.count
        rooto.count += 1
        rooto._ensure_pool(idx + 1)
        retv = _NodeCursor(rooto, idx, rect, 0, 0)
        retv._save_back()
        return retv

    @classmethod
    def create_with_children(cls, children, rooto):
        """
        给所有的children节点创建一个父节点(_NodeCursor对象)
        :param children:
        :param rooto:
        :return:
        """
        rect = union_all([child for child in children])
        assert not rect.swapped_x
        nc = _NodeCursor.create(rooto, rect)
        nc._set_children(children)
        assert not nc.is_leaf()
        return nc

    @classmethod
    def create_leaf(cls, rooto, leaf_obj, leaf_rect):
        """
        创建一个叶子节点，即创建一个_NodeCursor对象，rooto为RTree，
        将创建的叶子对象数据保存在rooto中
        :param rooto:
        :param leaf_obj:
        :param leaf_rect:
        :return:
        """
        rect = Rectangle(leaf_rect.x, leaf_rect.y, leaf_rect.xx, leaf_rect.yy)
        rect.swapped_x = True  # Mark as leaf by setting the xswap flag.
        res = _NodeCursor.create(rooto, rect)
        idx = res.index
        # 后面holds_leaves会进行判断，如果first_child == 0 则
        res.first_child = rooto.leaf_count
        rooto.leaf_count += 1
        rooto.leaf_pool.append(leaf_obj)
        res._save_back()
        # just for test
        res._become(idx)
        assert res.is_leaf()
        return res

    def __init__(self, rooto, index, rect, first_child, next_sibling):
        self.root = rooto
        self.rpool = rooto.rect_pool
        self.npool = rooto.node_pool

        self.index = index
        self.rect = rect
        self.next_sibling = next_sibling
        self.first_child = first_child

    def walk(self, predicate):
        if predicate(self):
            yield self
            if not self.is_leaf():
                for child in self.children():
                    for cr in child.walk(predicate):
                        yield cr

    def query_rect(self, r):
        """
        查询所有与r相交的矩形
        :param r:
        :return:
        """
        def p(o):
            return r.does_intersect(o.rect)

        for rr in self.walk(p):
            yield rr

    def query_point(self, point):
        """
        查询所有包含point的矩形
        :param point:
        :return:
        """
        def p(o):
            return o.rect.does_contain_point(point)

        for rr in self.walk(p):
            yield rr

    def lift(self):
        """
        获取一个_NodeCursor对象
        :return:
        """
        return _NodeCursor(self.root,
                           self.index,
                           self.rect,
                           self.first_child,
                           self.next_sibling)

    def _become(self, index):
        """
        将某个节点转换成对应需要转换成的节点，只需要改变以下，其中2，3，4数据需要通过index来retrieve
            1. index：要转换的节点在RTree中的索引
            2. first_child: 第一个孩子节点，只有在有孩子节点才有用，对于叶子节点是没有用的
            3. next_sibling: 下一个兄弟节点
            4. rect: 要转换成的节点的矩形数据，如果是叶子节点的话，swapped_x 为True
        :param index: 要转换的节点在RTree中的索引
        :return:
        """
        recti = index * 4
        nodei = index * 2
        rp = self.rpool
        x = rp[recti]
        y = rp[recti + 1]
        xx = rp[recti + 2]
        yy = rp[recti + 3]

        if x == 0.0 and y == 0.0 and xx == 0.0 and yy == 0.0:
            self.rect = NullRect
        else:
            self.rect = Rectangle(x, y, xx, yy)

        self.next_sibling = self.npool[nodei]
        self.first_child = self.npool[nodei + 1]
        self.index = index

    def is_leaf(self):
        """
        叶子节点的x,xx位置会进行交换，由此判断是否叶子节点
        :return:
        """
        return self.rect.swapped_x

    def has_children(self):
        return not self.is_leaf() and self.first_child != 0

    def holds_leaves(self):
        """
        判断该节点是否会包含叶子节点，也就是说是否是最低一层次的非叶子节点
        :return:
        """
        # first_child为0这种节点是还可以加入叶子节点或者非叶子结点的非叶子节点节点
        if self.first_child == 0:
            return True
        # 如果包含children而且children是叶子节点的话，为True
        return self.has_children() and self.get_first_child().is_leaf()

    def get_first_child(self):
        """
        获取第一个子节点，父节点的first_child记录了第一个孩子节点的位置
        因此可以在对应的rect_pool和node_pool找到相应的数据
        :return:
        """
        c = _NodeCursor(self.root, 0, NullRect, 0, 0)
        c._become(self.first_child)
        return c

    def leaf_obj(self):
        """
        获取第一个叶子节点的数据
        :return:
        """
        if self.is_leaf():
            return self.root.leaf_pool[self.first_child]
        return None

    def _save_back(self):
        """
        将当前节点的数据写入到rect_pool和node_pool中
        :return:
        """
        rp = self.rpool
        recti = self.index * 4
        nodei = self.index * 2

        if self.rect is not NullRect:
            self.rect.write_raw_coords(rp, recti)
        else:
            rp[recti] = 0
            rp[recti + 1] = 0
            rp[recti + 2] = 0
            rp[recti + 3] = 0

        self.npool[nodei] = self.next_sibling
        self.npool[nodei + 1] = self.first_child

    def children_count(self):
        count = 0
        for _ in self.children():
            count += 1
        return count

    def insert(self, leaf_obj, leaf_rect):  # pylint:disable=too-many-locals
        """
        向RTree中插入一个叶子节点对象和矩形框数据
        :param leaf_obj:
        :param leaf_rect:
        :return:
        """
        index = self.index
        while True:
            if self.holds_leaves():
                # 合并包裹众多叶子结点矩形框的大矩形框
                self.rect = self.rect.union(leaf_rect)
                self._insert_child(_NodeCursor.create_leaf(self.root, leaf_obj, leaf_rect))
                # 判断是否进行节点分裂
                self._balance()
                # 完成之后切换为最开始的节点
                self._become(index)
                return
            else:
                child = None
                min_area = -1.0
                # 找到能合并的，并且合并之后面积最小的子节点
                for _child in self.children():
                    x, y, xx, yy = _child.rect.coords()
                    lx, ly, lxx, lyy = leaf_rect.coords()
                    nx = x if x < lx else lx
                    nxx = xx if xx > lxx else lxx
                    ny = y if y < ly else ly
                    nyy = yy if yy > lyy else lyy
                    area = (nxx - nx) * (nyy - ny)
                    if min_area < 0 or area < min_area:
                        min_area = area
                        child = _child.index
                self.rect = self.rect.union(leaf_rect)
                self._save_back()
                self._become(child)  # recurse.

    def _balance(self):
        """
        判断是否进行节点分裂，如果需要进行分裂，则进行以下操作：
            1. 分别对需要进行节点分裂的子节点进行K-Means聚类
            2. 确定最优聚类
            3. 按照最优聚类的结果将节点分为同等数量的子节点，并且将原来的子节点按照聚类结果分别挂靠在新的子节点上
            4. 将新的子节点挂靠在以前的原始节点上
        :return:
        """
        if self.children_count() <= MAX_CHILDREN:
            return
        s_children = [child.lift() for child in self.children()]
        memo = {}
        # 将叶子节点分别分为2,3,4,5个中心点的cluster
        clusters = [k_means_cluster(k, s_children) for k in range(2, MAX_KMEANS)]
        score, best_cluster = max([(silhouette_coeff(cluster, memo), cluster) for cluster in clusters])

        # best_cluster是某个(n个中心点)的不同叶子节点的聚类，c是每个中心点所包含的所有叶子结点集合
        # 将不同聚类的叶子结点分别创建根节点，最后绑定到当前非叶子结点上
        nodes = [_NodeCursor.create_with_children(cluster, self.root) for cluster in best_cluster if len(cluster) > 0]
        self._set_children(nodes)

    def _set_children(self, children):
        """
        将所有children(_NodeCursor对象)置于当前的节点(_NodeCursor对象)下，
        并且每个child都有在RTree中的唯一索引index
        父节点通过first_child索引到第一个子节点，同时子节点通过next_sibling索引到其他子节点
        :param children:
        :return:
        """
        self.first_child = 0

        if not children:
            return

        prev = children[0]
        self.first_child = prev.index
        for child in children[1:]:
            prev.next_sibling = child.index
            prev._save_back()
            prev = child
        # last child
        prev.next_sibling = 0
        prev._save_back()
        self._save_back()

    def _insert_child(self, child):
        """
        向RTree中插入新的子节点
        :param child:
        :return:
        """
        # 原始节点的first_child作为插入叶子节点的兄弟节点
        child.next_sibling = self.first_child
        # 新插入的作为first_child
        self.first_child = child.index
        # 将数据写入rect_pool和node_pool
        child._save_back()
        self._save_back()

    def children(self):
        """
        获取当前节点的(_NodeCursor对象)的所有子节点
        :return:
        """
        if self.first_child == 0:
            return

        idx = self.index
        fc = self.first_child
        ns = self.next_sibling
        rect = self.rect

        self._become(self.first_child)
        while True:
            yield self
            if self.next_sibling == 0:
                break
            else:
                self._become(self.next_sibling)

        # Go back to becoming the same node we were.
        # self._become(idx)
        self.index = idx
        self.first_child = fc
        self.next_sibling = ns
        self.rect = rect


def avg_diagonals(node, onodes, memo_tab):
    nidx = node.index
    sv = 0.0
    for onode in onodes:
        k1 = (nidx, onode.index)
        k2 = (onode.index, nidx)
        if k1 in memo_tab:
            diag = memo_tab[k1]
        elif k2 in memo_tab:
            diag = memo_tab[k2]
        else:
            diag = node.rect.union(onode.rect).diagonal()
            memo_tab[k1] = diag
        sv += diag
    return sv / len(onodes)


def silhouette_w(node, cluster, next_closest_cluster, memo):
    ndist = avg_diagonals(node, cluster, memo)
    sdist = avg_diagonals(node, next_closest_cluster, memo)
    return (sdist - ndist) / max(sdist, ndist)


def silhouette_coeff(clustering, memo_tab):
    """
    轮廓系数，聚类效果好坏的一种评价方式
    :param clustering:
    :param memo_tab:
    :return:
    """
    # special case for a clustering of 1.0
    if len(clustering) == 1:
        return 1.0

    coeffs = []
    for cluster in clustering:
        others = [c for c in clustering if c is not cluster]
        others_cntr = [center_of_gravity(c) for c in others]
        ws = [silhouette_w(node, cluster, others[closest(others_cntr, node)], memo_tab) for node in cluster]
        cluster_coeff = sum(ws) / len(ws)
        coeffs.append(cluster_coeff)
    return sum(coeffs) / len(coeffs)


def center_of_gravity(nodes):
    """
    计算一堆nodes的中心点
    :param nodes:
    :return:
    """
    total_area = 0.0
    xs, ys = 0, 0
    for n in nodes:
        if n.rect is not NullRect:
            x, y, w, h = n.rect.extent()
            a = w * h
            # a是对x, y改变的权重
            xs = xs + (a * (x + (0.5 * w)))
            ys = ys + (a * (y + (0.5 * h)))
            total_area = total_area + a
    return xs / total_area, ys / total_area


def closest(centroids, node):
    """
    找到离某个node最近的中心点的索引
    :param centroids:
    :param node:
    :return:
    """
    x, y = center_of_gravity([node])
    dist = -1
    ridx = -1

    for (idx, (xx, yy)) in enumerate(centroids):
        # 计算距离
        dsq = ((xx - x) ** 2) + ((yy - y) ** 2)
        if -1 == dist or dsq < dist:
            dist = dsq
            ridx = idx
    return ridx


def k_means_cluster(k_group, nodes):
    """
    将nodes分为k组，每一组都最靠近某个中心点
    :param k_group: 将nodes分为k组
    :param nodes: 某个非叶子节点下的所有节点
    :return:
    """
    if len(nodes) <= k_group:
        return [[node] for node in nodes]

    nodes = list(nodes)
    # Initialize: take n random nodes.
    random.shuffle(nodes)
    # 找到k个中心点
    cluster_centers = [center_of_gravity([node]) for node in nodes[:k_group]]

    # Loop until stable:
    while True:
        clusters = [[] for _ in cluster_centers]

        for node in nodes:
            # 距离n节点最近的中心点，中心点有k个
            idx = closest(cluster_centers, node)
            # 按照不同的中心点将节点进行分类
            clusters[idx].append(node)

        clusters = [cluster for cluster in clusters if cluster]

        # 直到收敛
        new_cluster_centers = [center_of_gravity(cluster) for cluster in clusters]
        if new_cluster_centers == cluster_centers:
            return clusters
        else:
            cluster_centers = new_cluster_centers
