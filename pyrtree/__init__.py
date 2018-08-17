from __future__ import absolute_import

__all__ = ["rtree", "rect", "Rect", "RTree"]

from . import rect
from . import rtree

Rect = rect.Rect
RTree = rtree.RTree

"""
所以逐个的矩形框来创建RTree，在加入树的时候，
判断与当前非叶子节点中的其他叶子节点中是否有重叠
重叠的进行合并(如果找到多个叶子节点，那么取相交率最大的矩形(相交率=相交面积/总面积))，并且将值也进行合并
没有重叠的创建叶子节点，并且balance

如果在合并节点的时候和其他节点冲突的话，就略去该节点(扩张后的矩形如果和当前树中的矩形冲突，则不进行扩张)
"""
