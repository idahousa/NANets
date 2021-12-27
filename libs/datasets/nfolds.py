import math
import numpy as np
"""=================================================================================================================="""
"""Main class for single-class n-fold division. Donot use this class directly. Use the NFolds class instead."""
class sNFolds:
    def __init__(self, i_num_folds=3,i_num_samples=20):
        assert isinstance(i_num_folds,int)
        assert i_num_folds>=2
        assert isinstance(i_num_samples,int)
        assert i_num_samples>0
        self.num_folds   = i_num_folds
        self.num_samples = i_num_samples
        self.parts = []
        self.split()
    """Split a dataset (single class) into n-part for n-fold division"""
    def split(self):
        """1.Set seed for random"""
        np.random.seed(10)
        """2.Divide entire data into parts"""
        fold_size = math.ceil(self.num_samples/self.num_folds-0.5)
        samples = [i for i in range(self.num_samples)]
        for index in range(self.num_folds-1):
            s_samples = np.random.choice(samples,size=fold_size,replace=False)
            for sample in s_samples:
                samples.remove(sample)
            self.parts.append(s_samples)
        """The last part"""
        self.parts.append(samples)
    """Main function for train-val divisioin"""
    def __call__(self,i_fold_index=1,i_num_aug_samples=1):
        assert isinstance(i_fold_index,int)
        assert 0<i_fold_index<=self.num_folds
        """"Valid range is [0,num_folds]."""
        """As my design, fold index range is [1,, 2,...,num_folds]"""
        i_fold_index = i_fold_index - 1 #Convert from 0-based to 1-based indexing
        train_set = None
        for index in range(self.num_folds):
            if index == i_fold_index:#Taking validation set
                pass
            else:#Taking training set
                if train_set is None:
                    train_set = self.parts[index]
                else:
                    train_set = np.concatenate((train_set,self.parts[index]),axis=0)
        val_set = self.parts[i_fold_index]
        """Data augmentation for traininig dataset. 
           If i_num_samples<=len(train_set) then DONOT perform data augmentation"""
        aug_train = [item for item in train_set]
        if len(aug_train)>=i_num_aug_samples:
            pass
        else:
            gap = i_num_aug_samples - len(aug_train)
            while gap>0:
                rand_index = np.random.randint(0,len(aug_train))
                aug_train.append(aug_train[rand_index])
                gap -=1
        print('Number of train sample = ',len(train_set))
        print('Number of val   sample = ',len(val_set))
        train_set = np.array(aug_train)
        val_set   = np.array(val_set)
        return train_set,val_set
"""Main class for multi-class n-fold division. Please use this class as shown in the example usages"""
class NFolds:#Multiple Classes N-Fold
    def __init__(self,i_data=None,i_labels=None,i_num_fold=2, i_num_samples=None):
        assert isinstance(i_data,(list,tuple,np.ndarray))  #An mxn array of data
        assert isinstance(i_labels,(list,tuple,np.ndarray))#An (m,) array of labels. Ranging from 0 to k-1 where k is number of class
        assert isinstance(i_num_fold,int)
        assert i_num_fold>0
        self.num_folds  = i_num_fold
        self.num_labels = len(np.unique(i_labels))
        if isinstance(i_num_samples,(list,tuple)):
            assert self.num_labels == len(i_num_samples) #Number of samples in each class. Example [100,200,100]
            self.num_samples = i_num_samples
        else:
            self.num_samples = [0 for _ in range(self.num_labels)] #Taking original number of samples without performing data augmentation
        self.classes = [[] for _ in range(self.num_labels)]
        for index, item in enumerate(i_data):
            label = i_labels[index]
            assert isinstance(self.classes[label],(list,tuple))
            self.classes[label].append([index])
        self.sNFolds = [sNFolds(i_num_folds=i_num_fold,i_num_samples=len(self.classes[i])) for i in range(self.num_labels)]
        """Print infor"""
        class_sizes = [len(self.classes[i]) for i in range(self.num_labels)]
        print('class_sizes = ',class_sizes)
    """Main function for taking item indice accoridng to fold index"""
    def __call__(self,i_fold_index=1):
        assert isinstance(i_fold_index,int)
        assert 0<i_fold_index<=self.num_folds
        """"Check the sNFolds, if 0<i_fold_index<=self.num_folds, then the input range is [1,2,3,...,num_folds]
        Note, the order of fold is not matter.I used the fold in range
        [1,2,...,num_folds] is just for convenience."""
        train_sets,val_sets=[],[]
        for index in range(self.num_labels):
            """Use i_fold_index - 1 here to match with the start index of 0 in the sNFolds class"""
            train_set,val_set = self.sNFolds[index](i_fold_index=i_fold_index,i_num_aug_samples=self.num_samples[index])
            """Matching between local index to global index"""
            for i in train_set:
                train_sets.append(self.classes[index][i][0])#As my design
            for i in val_set:
                val_sets.append(self.classes[index][i][0])#As my design
        return train_sets,val_sets
"""==============================================================="""
if __name__ == "__main__":
    print('This module is for nfold division (randomly)')
    print('Multi class division')
    data    = np.random.randint(0,100,(100,3))
    labels  = np.random.randint(0,5,(100,))
    example = NFolds(i_data=data,i_labels=labels,i_num_fold=3)
    example_train_sets,example_val_sets = example(i_fold_index=1)
    print(np.sort(example_train_sets))
    print('Size of training sets: ',len(example_train_sets))
    print(np.sort(example_val_sets))
    print('Size of validation sets: ',len(example_val_sets))
    print('='*100)
    print('Single class division')
    sexample = sNFolds(i_num_folds=3, i_num_samples=100)
    sexample_train_sets, sexample_val_sets = sexample(i_fold_index=1,i_num_aug_samples=100)
    print(np.sort(sexample_train_sets))
    print('Size of training sets: ', len(sexample_train_sets))
    print(np.sort(sexample_val_sets))
    print('Size of validation sets: ', len(sexample_val_sets))
    print('=' * 100)
"""==============================================================="""