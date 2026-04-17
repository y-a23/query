# 二叉树最大路径和

#  1
# 2 3

#     -10
#  9          20
# null null 15   7


#    -10
#  9      20
# 30 30  15  17
class Node:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right

class solution:
    def __init__(self):
        self.ans = 0 
    
    def dfs(self, root):
        if root == None:
            return 0
        leftmax = self.dfs(root.left)
        rightmax = self.dfs(root.right)
        self.ans = max(self.ans, max(leftmax, 0) + max(rightmax, 0) + root.val)
        return root.val + max(0, max(leftmax, rightmax))
    
    def getans(self, root):
        self.ans = root.val
        self.dfs(root)
        return self.ans

def test1():
    root = Node(1, None, None)
    root.left = Node(2, None, None)
    root.right = Node(3, None, None)
    tmp = solution()
    print(tmp.getans(root))

def test2():
    root = Node(-10, None, None)
    root.left = Node(9, None, None)

    root.left.left = Node(30, None, None)
    root.left.right = Node(30, None, None)


    root.right = Node(20, None, None)
    root.right.left = Node(15, None, None)
    root.right.right = Node(7, None, None)
    tmp = solution()
    print(tmp.getans(root))

test1()
test2()

    