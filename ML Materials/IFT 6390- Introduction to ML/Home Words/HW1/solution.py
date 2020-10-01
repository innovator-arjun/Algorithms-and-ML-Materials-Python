import numpy as np

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, banknote):
        features=banknote[:,:4]
        return np.mean(features,axis=0)

    def covariance_matrix(self, banknote):
        features=banknote[:,:4]
        return np.cov(features.T)

    def feature_means_class_1(self, banknote):
        data_class_1 = banknote[banknote[:,4] == 1, :4]
        return np.mean(data_class_1,axis=0)

    def covariance_matrix_class_1(self, banknote):
        data_class_1 = banknote[banknote[:,4] == 1, :4]
        return np.cov(data_class_1.T)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.label_length=len(np.unique(train_labels))
        self.train_inputs=train_inputs
        self.train_labels=train_labels

    def compute_predictions(self, test_data):
        neighbours=[]
        
        length=test_data.shape[0]
        classes_pred=np.zeros(length)
        
        counts=np.ones((length,self.label_length))

        radius=self.h
        for i in range(0,len(test_data)):
            
            distance=(np.sum((np.abs(test_data[i] - self.train_inputs)) ** 2, axis=1)) ** (1.0 / 2)
            neighbours = np.array([j for j in range(len(distance)) if distance[j] < radius])
            for k in neighbours:
                counts[i,int(self.train_labels[k])]+=1
            if max(counts[i,:])==1:
                classes_pred[i]=draw_rand_label(test_data[i],self.label_list)
            else:
                classes_pred[i] = np.argmax(counts[i, :])
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.label_list = np.unique(train_labels)
        self.label_length=len(np.unique(train_labels))
        self.train_inputs=train_inputs
        self.train_labels=train_labels

    def compute_predictions(self, test_data):
        neighbours=[]
        
        length=test_data.shape[0]
        classes_pred=np.zeros(length)
        counts=np.ones((length,self.label_length))
        for i in range(len(test_data)):
            distances = (np.sum((np.abs(test_data[i] - self.train_inputs)) ** 2, axis=1)) ** (1.0 / 2)
            dic = dict.fromkeys(self.label_list, 1)
            for j in range(len(distances)):
                sig=(1/(np.sqrt(2*np.pi)*self.sigma))*np.exp(-(distances[j]**2/(2*self.sigma**2)))
                dic[self.train_labels[j]]+=sig
            classes_pred[i]= max(dic, key=dic.get)
        return classes_pred


def split_dataset(banknote):
    length=list(range(0,len(banknote)))
    train_index=[]
    validation_index=[]
    test_index=[]
    
    for i in range(0,len(length)):
        if i%5==0 or i%5==1 or i%5==2:
            train_index.append(i)
        elif i%5==3:
            validation_index.append(i)
        elif i%5==4:
            test_index.append(i)
    train=banknote[train_index]
    validation=banknote[validation_index]
    test=banknote[test_index]
    return (train,validation,test)


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        knn_hard=HardParzen(h)
        knn_hard.train(self.x_train,self.y_train)
        knn_hard_pred=knn_hard.compute_predictions(self.x_val)

        n_classes = len(np.unique(knn_hard_pred))
        total_correct=0
        for i in range(0,len(knn_hard_pred)):
            if knn_hard_pred[i]==self.y_val[i]:
                total_correct+=1       
        total_pred=len(knn_hard_pred)        
        
        return 1-((float(total_correct) / float(total_pred)))

    def soft_parzen(self, sigma):
        knn_soft=SoftRBFParzen(sigma)
        knn_soft.train(self.x_train,self.y_train)
        knn_soft_pred=knn_soft.compute_predictions(self.x_val)
        correct=0
        for j in range(0,len(knn_soft_pred)):
            if knn_soft_pred[j]==self.y_val[j]:
                correct+=1
        total_pred=len(knn_soft_pred) 
        return 1-((float(correct) / float(total_pred)))


def get_test_errors(banknote):
    split_dataset(banknote)
    dataset=split_dataset(banknote)
    test=dataset[2]
    test_label=test[:,4]

    train=dataset[0]
    train_label=train[:,4]
    error=ErrorRate(train[:,:4],train_label,test[:,:4],test_label)
    h=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    h_error={}
    for i in range(len(h)):
        h_error[h[i]]= error.hard_parzen(h[i])
            
    sigma=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 3.0, 10.0, 20.0]
    s_error={}
    for i in range(len(sigma)):
        s_error[sigma[i]]= error.soft_parzen(sigma[i])
    soft_min=min(s_error,key=s_error.get)
    hard_min=min(h_error,key=h_error.get)
#     res=np.array([
#         [hard_min,h_error[hard_min]],
#         [soft_min,s_error[soft_min]]
#     ])

    return np.array([h_error[hard_min],s_error[hard_min] ])


def random_projections(X, A):
    res=X.dot(A)
    res=res/np.sqrt(2)
    return res