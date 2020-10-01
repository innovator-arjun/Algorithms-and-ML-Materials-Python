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
        self.num_classes=len(np.unique(train_labels))
        self.train_inputs=train_inputs
        self.train_labels=train_labels
    def compute_predictions(self, test_data):
        neighbours=[]
        
        length=test_data.shape[0]
        classes_pred=np.zeros(length)
        counts=np.ones((length,self.num_classes))
        radius=self.h
        for i in range(len(test_data)):
            
            distance=(np.sum((np.abs(test_data[i] - self.train_inputs)) ** 2, axis=1)) ** (1.0 / 2)
            neighbours = np.array([j for j in range(len(distance)) if distance[j] < radius])
            if len(neighbours)==0:
                classes_pred[i]=draw_rand_label(test_data[i],self.label_list)
            else:
                for k in neighbours:
                    counts[i,int(self.train_labels[k])]+=1

            classes_pred[i] = np.argmax(counts[i, :])
        return classes_pred.reshape(-1,1)


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        # self.label_list = np.unique(train_labels)
        pass

    def compute_predictions(self, test_data):
        pass


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
#     print(len(train),len(validation),len(test))
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
        
        return (float(total_correct) / float(total_pred))

    def soft_parzen(self, sigma):
        pass


def get_test_errors(banknote):
    pass


def random_projections(X, A):
    pass