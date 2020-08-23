class Node:
    def __init__(self, data, parent):
        self.data = data
        self.leftChild = None
        self.rightChild = None
        self.parent = parent


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, data):

        if self.root is None:
            self.root = Node(data, None)
        else:
            self.insert_node(data, self.root)

    def insert_node(self, data, node):
        # We have to go to the left subtree
        if data < node.data:
            if node.leftChild:
                self.insert_node(data, node.leftChild)  # Recurssion
            else:
                node.leftChild = Node(data, node)
        # we have to visit the right sub tree
        else:
            if node.rightChild:
                self.insert_node(data, node.rightChild)
            else:
                node.rightChild = Node(data, node)

    def traverse(self):
        if self.root is not None:
            self.traverse_inoder(self.root)

    def traverse_inoder(self, node):
        if node.leftChild:
            self.traverse_inoder(node.leftChild)
        print(node.data)

        if node.rightChild:
            self.traverse_inoder(node.rightChild)


bst = BinarySearchTree()
bst.insert(10)
bst.insert(5)
bst.insert(-5)
bst.insert(1)
bst.insert(99)
bst.insert(34)


bst.traverse()
