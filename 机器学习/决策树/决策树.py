#!user/bin/python

class noteModel:
    def __init__(self,Id,value,fatherId):
        self.Id=Id
        self.value=value
        self.fatherId=fatherId
        self.children = []
 
    def addChild(self,*child):
        self.children += child

    def printTree(self,layer):
        map(lambda child:child.printTree(layer + 1), self.children)
        print ('  '*layer + self.value)

    def creat(self, *list):
        # 循环列表，绑定父子关系，形成一个树
        for i in range(0, len(list)):
            for j in range(0, len(list)):
                if list[j].fatherId == list[i].Id:
                    list[i].addChild(list[j])

class Tree:

    def __init__(self, *list):
        # 查询数据库，并生成列表
        # list = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10]
        pass



    def printTree(self):
        pass

def main():

   #数据表模拟，数据库有 Id, value, fatherId 三个字段，t1-t10代表10条数据行
    t1 = noteModel(1,'A-1',0)
    t2 = noteModel(2,'B-1',1)
    t3 = noteModel(3,'B-2',1)
    t4 = noteModel(4,'C-1',2)
    t5 = noteModel(5,'C-2',2)
    t6 = noteModel(6,'C-3',3)
    t7 = noteModel(7,'C-4',3)
    t8 = noteModel(8,'D-1',4)
    t9 = noteModel(9,'E-1',8)
    t10 = noteModel(10,'E-2',8)



    #打印树
    t1.printTree(0)
    t2.printTree(1)
    t3.printTree(1)
    t4.printTree(2)
    t5.printTree(2)
    t6.printTree(3)
    t7.printTree(3)
    t8.printTree(4)
    t9.printTree(8)
    t1.printTree(8)
if __name__ == '__main__':
    main()