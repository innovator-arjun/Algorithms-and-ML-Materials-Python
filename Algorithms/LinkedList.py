class Node: # To create a new node with data and address to point the next Node
    def __init__(self,data):# Constructor to initalize
        self.data=data
        self.next=None

# Class for linked list operations linked list
class LinkedListImp:
    def __init__(self):
        self.head=None
        self.counter=0

#This is why love linked list 0(1) complexity
    def insert_beg(self,data):
        newNode=Node(data)
        self.counter+=1
        if not self.head:
            self.head=newNode
        else:
            newNode.next=self.head
            self.head=newNode



#O(N) Complexity
    def insert_end(self,data):
        self.counter+=1
        newNode=Node(data)
        actual=self.head


        while actual.next is not None:
            #print(actual.data)
            actual=actual.next

        actual.next=newNode

#O(1) Complexity, we just have to return only
    def size_of_linked_list(self):
        print("The size is",self.counter)

# O(N) Complexity, as we want to traverse the entire linkedlist

    def traverse(self):
        actual=self.head
        while actual is not None:
            print(actual.data)
            actual=actual.next

    def remove(self,data):
        if not self.head:
            return
        actual_node=self.head
        previous_node=None
        while actual_node is not None and actual_node.data!=data:
            previous_node=actual_node
            actual_node=actual_node.next

        if actual_node is None: # search miss -the item to remove is not in the list:
            return
        self.counter-=1 # To decrement the number of node
        if previous_node is None:# the head node we want to remove, update the head node
            self.head=actual_node.next
        else: # remove the intermediate node
            previous_node.next=actual_node.next


linkedlist=LinkedListImp()
linkedlist.insert_beg(4)
linkedlist.insert_beg(3)
linkedlist.insert_beg(7)
linkedlist.insert_end(100)

linkedlist.traverse()
print("---------")
linkedlist.remove(3)
linkedlist.traverse()

print('--------')
linkedlist.size_of_linked_list()




