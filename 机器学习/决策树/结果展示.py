# -*- coding: utf-8 -*-

# @File    : graphviz_demo.py
# @Date    : 2018-07-03
# @Author  : Peng Shiyu

from graphviz import Digraph

dot = Digraph(comment='The Test Table', format="png")

# 添加圆点A,A的标签是Dot A
dot.node('A', 'root_data:labels')

# 添加圆点 B, B的标签是Dot B
dot.node('B', 'child_data_1')
dot.node('C', 'child_data_2')

dot.node('D', 'child_child_data_1_1')
dot.node('E', 'child_child_data_1_2')

dot.node('F', 'child_child_data_2_1')
dot.node('G', 'child_child_data_2_2')

# dot.view()

# 添加圆点 C, C的标签是Dot C
# dot.node('C', 'Dot C')
# dot.view()

# 创建一堆边，即连接AB的两条边，连接AC的一条边。
dot.edges(['AB', 'AC'])
dot.edges(['BD', 'BE'])
dot.edges(['CF', 'CG'])
# dot.view()

class tree:

    def __init__(self, root, leaf):
        dot.node('a', root)

        for i in range(len(leaf)):

            node_name = ['a','b']
            dot.node(node_name[i], leaf[i])
            dot.edges(['a'+ node_name[i]])

tree = tree(root='root_data:labels', leaf=['child_data_1','child_data_2'])

# 在创建两圆点之间创建一条边
# dot.edge('B', 'C', 'test')
# dot.view()

# 保存source到文件，并提供Graphviz引擎
dot.save('test-table.gv')  # 保存
dot.render('test-table.gv')
# dot.view()  # 显示

from graphviz import Source

s = Source.from_file('test-table.gv')
print(s.source)  # 打印代码
# s.view()  # 显示