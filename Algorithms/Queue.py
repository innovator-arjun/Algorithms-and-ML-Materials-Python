#FIFO- First In First out- Queue

class Queue:

    def __init__(self):
        self.queue=[]

    # O(1) running time, to check if the queue is empty
    def isEmpty(self):
        return self.queue==[]

    #O(1), to insert the element in the queue
    def enqueue(self,data):
        self.queue.append(data)

    #0(N), when the first element is deleted the whole array to be shifted left. Hence we use linkedlist

    def dequeue(self):
        if self.size()<1:
            return "No item to delete"
        data=self.queue[0]
        self.queue.remove(data)
        return data
    #O(1), time complexity
    def peek(self):
        return self.queue[0]

    #O(1) constant running time
    def size(self):
        return len(self.queue)

queue=Queue()
queue.enqueue(10)

queue.enqueue(20)

queue.enqueue(30)
print(queue.dequeue())
print(queue.size())
print(queue.dequeue())

print(queue.dequeue())

print(queue.dequeue())

