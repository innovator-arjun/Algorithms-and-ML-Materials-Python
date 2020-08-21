#LIFO- Last In First Out -- Stack Implementation

class Stack:

    def __init__(self):
        self.stack=[]

    #Push-Insert item into the stack O(1)
    def push(self,data):
        self.stack.append(data)

    # Pop- Remove and return the last item we have inserted - O(1)
    def pop(self):
        if self.size()<1:
            return "No element to delete"
        data=self.stack[-1] # To get the last element in the array/list
        self.stack.remove(data)
        return data

    #It removes the last item with out removing it  // O(1)
    def peek(self):
        return self.stack[-1]

    # To check if the stack is empty or not
    def is_empty(self):
        return self.stack==[]
    # To get the size of the stack
    def size(self):
        return len(self.stack)

stack=Stack()
stack.push(10)
stack.push(20)
stack.push(30)
print("The popped item is ",stack.pop())
print("Size is", stack.size())

print("The top element is",stack.peek())
print(stack.pop())
print(stack.pop())
print(stack.pop())

print("Size is", stack.size())

