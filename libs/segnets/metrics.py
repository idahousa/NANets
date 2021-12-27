import numpy as np
from libs.logs import Logs
"""=================================================================================================================="""
class SegMetrics:
    sm = 1e-10  # Smoothing factor for measuring dice and jaccard score
    def __init__(self,i_num_classes=2,i_care_background=False):
        assert isinstance(i_num_classes,int)     #Total of classes including background
        assert isinstance(i_care_background,bool)#Flag to decide whether consider background or not.
        assert i_num_classes>=1 #Note: the 0 is for background
        self.TP = 0.  # Positive => Positive
        self.FP = 0.  # Negative => Positive
        self.TN = 0.  # Negative => Negative
        self.FN = 0.  # Positive => Negative
        self.num_classes       = i_num_classes
        self.init_object_index = 0 if i_care_background else 1
    def erase_indicator(self):
        self.TP = 0.  # Positive => Positive
        self.FP = 0.  # Negative => Positive
        self.TN = 0.  # Negative => Negative
        self.FN = 0.  # Positive => Negative
    """The similarity of two sets. Also named as F1_Score"""
    @classmethod
    def get_dice(cls,i_TP=0.0,i_FP=0.0,i_FN=0.0):
        """If both output and target are empty, it makes sure dice is 1."""
        assert isinstance(i_TP,float)
        assert isinstance(i_FP,float)
        assert isinstance(i_FN,float)
        assert i_TP >= 0.0
        assert i_FP >= 0.0
        assert i_FN >= 0.0
        """DICE = 2*JACCARD/(JACCARD + 1)"""
        norminator  = 2 * i_TP
        denominator = 2 * i_TP + i_FP + i_FN
        dice = (norminator + cls.sm) / (denominator + cls.sm)
        return dice * 100.
    @classmethod
    def get_Jaccard(cls,i_TP=0.0,i_FP=0.0,i_FN=0.0):
        """If both output and target are empty, it makes sure dice is 1."""
        assert isinstance(i_TP,float)
        assert isinstance(i_FP,float)
        assert isinstance(i_FN,float)
        assert i_TP >= 0.0
        assert i_FP >= 0.0
        assert i_FN >= 0.0
        """JACCARD = DICE / (2- DICE)"""
        norminator  = i_TP
        denominator = i_TP + i_FP + i_FN
        jaccard     = (norminator + cls.sm) / (denominator + cls.sm)
        return jaccard * 100.
    """Accuracy of the foreground detection/segmentation respect to ground-truth label"""
    """The precision is intuitively the ability of the classifier not to label as positive a sample that is negative."""
    @staticmethod
    def get_precision(i_TP=0.0,i_FP=0.0):
        """Precision answers the following: How many of those who we labeled as disease are actually disease?"""
        assert isinstance(i_TP,float)
        assert isinstance(i_FP,float)
        assert i_TP >= 0.0
        assert i_FP >= 0.0
        norminator  = i_TP
        denominator = i_TP + i_FP
        if denominator > 0:
            precision = norminator / denominator
        else:  # Exchange background <-> foreground => FP = 0 (In sklearn, this is an undefined case)
            precision = 1.0  # precision = 0.0
        return precision * 100.0
    """Accuracy of the foreground detection/segmentation respect to detection/segmentation accuracy"""
    """The recall is intuitively the ability of the classifier to find all the positive samples."""
    @staticmethod
    def get_recall(i_TP=0.0,i_FN=0.0):
        """Note: Recall = Sensitivity"""
        assert isinstance(i_TP,float)
        assert isinstance(i_FN,float)
        assert i_TP >= 0.0
        assert i_FN >= 0.0
        norminator  = i_TP
        denominator = i_TP + i_FN
        if denominator > 0:
            recall = norminator / denominator
        else:  # Exchange background <-> foreground => FN = 0 (In sklearn, this is an undefined case)
            recall = 1.0  # recall = 0.0
        return recall * 100.
    @staticmethod
    def get_specificity(i_TN=0.0,i_FP=0.0):
        """Specifity answers the following question: Of all the people who are healthy, how many of those did we correctly predict?"""
        assert isinstance(i_TN,float)
        assert isinstance(i_FP,float)
        assert i_TN >= 0.0
        assert i_FP >= 0.0
        norminator  = i_TN
        denominator = i_TN + i_FP
        if denominator > 0:
            specificity = norminator / denominator
        else:
            specificity = 1.0
        return specificity * 100
    """Overall accuracy"""
    @staticmethod
    def get_accuracy(i_TP=0.0,i_TN=0.0,i_FP=0.0,i_FN=0.0):
        assert isinstance(i_TP,float)
        assert isinstance(i_TN,float)
        assert isinstance(i_FP,float)
        assert isinstance(i_FN,float)
        assert i_TP>=0.0
        assert i_TN>=0.0
        assert i_FP>=0.0
        assert i_FN>=0.0
        assert (i_TP + i_FN + i_TN + i_FP)>0
        acc = (i_TP + i_TN) / (i_TP + i_FN + i_TN + i_FP)
        return acc * 100.0
    """Get indicators"""
    """Count the TP,TN,FP,and FN globally"""
    @staticmethod
    def get_indicator(i_label=None,i_pred=None,i_object_index=1):
        assert isinstance(i_label,np.ndarray)
        assert isinstance(i_pred,np.ndarray)
        assert isinstance(i_object_index,int)
        assert i_object_index>=0
        assert len(i_label.shape)==len(i_pred.shape)
        cmps = SegMetrics.compare(i_label.shape,i_pred.shape)
        assert sum(cmps)==0
        label = (i_label == i_object_index).astype(np.int)
        pred  = (i_pred  == i_object_index).astype(np.int)
        mask  = label*2+ pred + 1
        """Calculation of TP,TN,FP,and FN"""
        TN = np.sum(mask[mask == 1])/1.0  # 0 --> 0
        FP = np.sum(mask[mask == 2])/2.0  # 0 --> 1
        FN = np.sum(mask[mask == 3])/3.0  # 1 --> 0
        TP = np.sum(mask[mask == 4])/4.0  # 1 --> 1
        #print('TN = {}, TP = {}, FN = {}, FP = {}'.format(TN,TP,FN,FP))
        return TN,FP,TP,FN
    def get_indicators(self,i_label=None,i_pred=None):
        assert isinstance(i_label, np.ndarray)
        assert isinstance(i_pred, np.ndarray)
        assert len(i_label.shape) == len(i_pred.shape)
        cmps = self.compare(i_label.shape, i_pred.shape)
        assert sum(cmps) == 0
        for index in range(self.init_object_index,self.num_classes):
            TN, FP, TP, FN = self.get_indicator(i_label=i_label,i_pred=i_pred,i_object_index=index)
            self.TN += TN
            self.TP += TP
            self.FN += FN
            self.FP += FP
        return True
    """Support functions"""
    @classmethod
    def compare(cls, i_x=None, i_y=None):
        """Compare two list or tuple of number"""
        assert isinstance(i_x, (list, tuple))
        assert isinstance(i_y, (list, tuple))
        assert len(i_x) == len(i_y)
        cmps = []
        for index, x in enumerate(i_x):
            y = i_y[index]
            if x == y:
                cmps.append(0)
            else:
                cmps.append(1)
        """cmps serves as register that indicates the similarity between two lists or tuples"""
        return cmps
    @staticmethod
    def get_mean(i_measures=None):
        """Custom for measurement of mean"""
        assert isinstance(i_measures, np.ndarray)
        assert len(i_measures.shape) == 2
        mean_measures = np.mean(i_measures, axis=0)
        for index, item in enumerate(mean_measures):
            mean_measures[index] = round(item, 3)
        return mean_measures
    @staticmethod
    def get_std(i_measures=None):
        """Custom for measurement of std"""
        assert isinstance(i_measures, np.ndarray)
        assert len(i_measures.shape) == 2
        std_measures = np.std(i_measures, axis=0)
        for index, item in enumerate(std_measures):
            std_measures[index] = round(item, 3)
        return std_measures
    """Main function"""
    def get_metrics(self,i_label=None,i_pred=None):
        assert isinstance(i_label, np.ndarray)
        assert isinstance(i_pred, np.ndarray)
        label = np.squeeze(i_label)
        pred  = np.squeeze(i_pred)
        assert len(label.shape) == len(pred.shape),'Got values: {} vs {}'.format(label.shape,pred.shape)
        cmps = self.compare(label.shape, pred.shape)
        assert sum(cmps) == 0, 'Got values: {} vs {}'.format(label.shape,pred.shape)
        self.get_indicators(i_label=label,i_pred=pred)
        measures = list()
        measures.append(self.get_dice(i_TP=self.TP,i_FP=self.FP,i_FN=self.FN))                    # Dice
        measures.append(self.get_Jaccard(i_TP=self.TP,i_FP=self.FP,i_FN=self.FN))                 # Jaccard
        measures.append(self.get_precision(i_TP=self.TP,i_FP=self.FP))                            # Precision
        measures.append(self.get_recall(i_TP=self.TP,i_FN=self.FN))                               # Recall
        measures.append(self.get_recall(i_TP=self.TP,i_FN=self.FN))                               # Sensitivity = Recall
        measures.append(self.get_specificity(i_TN=self.TN,i_FP=self.FP))                          # Specificity
        measures.append(self.get_accuracy(i_TP=self.TP,i_TN=self.TN,i_FP=self.FP,i_FN=self.FN))   # Overall Accuracy
        self.erase_indicator()
        #print(measures)
        #plt.subplot(1,2,1)
        #plt.imshow(i_label)
        #plt.subplot(1,2,2)
        #plt.imshow(i_pred)
        #plt.show()
        return np.array(measures)
    def eval(self,i_labels=None,i_preds=None,i_object_care=False,i_tops=180):
        assert isinstance(i_tops,int)
        assert i_tops>=0
        """Evaluation of a dataset"""
        assert isinstance(i_labels,(list,tuple,np.ndarray))
        assert isinstance(i_preds,(list,tuple,np.ndarray))
        assert isinstance(i_object_care,bool) #Set to True to only consider the cases where object exists in input image. i.e. label is not ZERO
        if isinstance(i_labels,(list,tuple)):
            num_labels = len(i_labels)
        else:
            assert isinstance(i_labels,np.ndarray)
            num_labels = i_labels.shape[0]
        if isinstance(i_preds,(list,tuple)):
            num_preds  = len(i_preds)
        else:
            assert isinstance(i_preds,np.ndarray)
            num_preds  = i_preds.shape[0]
        assert num_labels == num_preds
        measures = list()
        for index, label in enumerate(i_labels):
            if i_object_care:
                if np.sum(label)>0:
                    measures.append(self.get_metrics(i_label=label,i_pred=i_preds[index]))
                else:
                    pass
            else:
                measures.append(self.get_metrics(i_label=label,i_pred=i_preds[index]))
        measures = np.array(measures)
        """Find the top most correct prediction results"""
        if i_tops>0:
            top_measures = []
            dices = measures[:,0].copy()
            while True:
                index = np.argmax(dices)
                dices[index] = 0
                top_measures.append(measures[index,:])
                if len(top_measures)>=i_tops:
                    break
                else:
                    continue
            top_measures = np.array(top_measures)
            print(top_measures.shape,self.get_mean(top_measures),self.get_std(top_measures))
        else:
            pass
        return measures,self.get_mean(measures),self.get_std(measures)
class SegMetrics3D:
    def __init__(self):
        self.TPs = 0.
        self.TNs = 0.
        self.FPs = 0.
        self.FNs = 0.
        self.preds  = []  #Use for measuring 2D performance
        self.labels = []  #Use for measuring 2D performance
    def erase_indicators(self):
        self.TPs = 0.  # Positive => Positive
        self.FPs = 0.  # Negative => Positive
        self.TNs = 0.  # Negative => Negative
        self.FNs = 0.  # Positive => Negative
        self.preds.clear()
        self.labels.clear()
    @classmethod
    def compare(cls, i_x=None, i_y=None):
        """Compare two list or tuple of number"""
        assert isinstance(i_x, (list, tuple))
        assert isinstance(i_y, (list, tuple))
        assert len(i_x) == len(i_y)
        cmps = []
        for index, x in enumerate(i_x):
            y = i_y[index]
            if x == y:
                cmps.append(0)
            else:
                cmps.append(1)
        """cmps serves as register that indicates the similarity between two lists or tuples"""
        return cmps
    def get_measures(self, i_labels=None,i_preds=None,i_object_index=1):
        """i_labels and i_preds are labels and preds of a 3D images"""
        assert isinstance(i_labels,np.ndarray)
        assert isinstance(i_preds,np.ndarray)
        assert isinstance(i_object_index,int)
        assert i_object_index>0
        if len(i_labels.shape)==3:
            i_labels = np.expand_dims(i_labels,axis=-1)
        else:
            assert len(i_labels.shape) == 4
            assert i_labels.shape[-1] == 1
        if len(i_preds.shape)==3:
            i_preds = np.expand_dims(i_preds,axis=-1)
        else:
            assert len(i_preds.shape)== 4
            assert i_preds.shape[-1] == 1
        assert len(i_labels.shape)==4 #None x Height x Width x Depth
        assert len(i_preds.shape)==4  #None x Height x Width x Depth
        assert np.sum(self.compare(i_x=i_labels.shape,i_y=i_preds.shape))==0
        TNs, FPs, TPs,FNs = 0, 0, 0, 0
        for index,label in enumerate(i_labels):
            pred = i_preds[index].copy()
            TN,FP,TP,FN = SegMetrics.get_indicator(i_label=label,i_pred=pred,i_object_index=i_object_index)
            #Logs.log('Single 2D : TP = {} - FP = {} - TN = {} - FN = {}'.format(TP, FP, TN, FN))
            TNs += TN
            FPs += FP
            TPs += TP
            FNs += FN
            self.preds.append(pred)
            self.labels.append(label)
        """Update to all dataset"""
        self.TPs += TPs
        self.TNs += TNs
        self.FPs += FPs
        self.FNs += FNs
        """Log details"""
        Logs.log('Single 3D        : TPs = {} - FPs = {} - TNs = {} - FNs = {}'.format(TPs, FPs, TNs, FNs))
        Logs.log('Current Global 3D: TPs = {} - FPs = {} - TNs = {} - FNs = {}'.format(self.TPs, self.FPs, self.TNs, self.FNs))
        """Measurement for single 3D image"""
        measures = list()
        measures.append(SegMetrics.get_dice(i_TP=TPs, i_FP=FPs, i_FN=FNs))                # Dice
        measures.append(SegMetrics.get_Jaccard(i_TP=TPs, i_FP=FPs, i_FN=FNs))             # Jaccard
        measures.append(SegMetrics.get_precision(i_TP=TPs, i_FP=FPs))                     # Precision
        measures.append(SegMetrics.get_recall(i_TP=TPs, i_FN=FNs))                        # Recall
        measures.append(SegMetrics.get_recall(i_TP=TPs, i_FN=FNs))                        # Sensitivity = Recall
        measures.append(SegMetrics.get_specificity(i_TN=TNs, i_FP=FPs))                   # Specificity
        measures.append(SegMetrics.get_accuracy(i_TP=TPs, i_TN=TNs, i_FP=FPs, i_FN=FNs))  # Overall Accuracy
        return measures
    def measures(self,i_labels=None,i_preds=None,i_object_index=1):
        assert isinstance(i_labels,(list,tuple))
        assert isinstance(i_preds,(list,tuple))
        assert isinstance(i_object_index,int)
        assert i_object_index>0 #Donot care background
        assert len(i_labels)==len(i_preds)
        measures = list()
        for index,label in enumerate(i_labels):
            pred = i_preds[index]
            measures.append(self.get_measures(i_labels=label,i_preds=pred,i_object_index=i_object_index))
        measures = np.array(measures)
        """Log details"""
        Logs.log('Global: TPs = {} - FPs = {} - TNs = {} - FNs = {}'.format(self.TPs,self.FPs,self.TNs,self.FNs))
        """Overall Measure (global)"""
        global_measures = list()
        global_measures.append(SegMetrics.get_dice(i_TP=self.TPs, i_FP=self.FPs, i_FN=self.FNs))     # Dice
        global_measures.append(SegMetrics.get_Jaccard(i_TP=self.TPs, i_FP=self.FPs, i_FN=self.FNs))  # Jaccard
        global_measures.append(SegMetrics.get_precision(i_TP=self.TPs, i_FP=self.FPs))               # Precision
        global_measures.append(SegMetrics.get_recall(i_TP=self.TPs, i_FN=self.FNs))                  # Recall
        global_measures.append(SegMetrics.get_recall(i_TP=self.TPs, i_FN=self.FNs))                  # Sensitivity = Recall
        global_measures.append(SegMetrics.get_specificity(i_TN=self.TNs, i_FP=self.FPs))             # Specificity
        global_measures.append(SegMetrics.get_accuracy(i_TP=self.TPs, i_TN=self.TNs, i_FP=self.FPs, i_FN=self.FNs))  # Overall Accuracy
        global_measures = np.array(global_measures)
        Logs.log('-' * 100)
        Logs.log_matrix(i_str='Global 3D measurement',i_matrix=np.expand_dims(global_measures,axis=0))
        Logs.log('-' * 100)
        """2D Performance measurement"""
        """Performance measurement"""
        evaluer = SegMetrics(i_num_classes=i_object_index+1, i_care_background=False)
        Logs.log('Using entire dataset')
        measures2d, measure_mean2d, measure_std2d = evaluer.eval(i_labels=self.labels, i_preds=self.preds, i_object_care=False)
        Logs.log('Measure shape (2D) = {}'.format(measures2d.shape))
        Logs.log('Measure mean  (2D) = {}'.format(measure_mean2d))
        Logs.log('Measure std   (2D) = {}'.format(measure_std2d))
        Logs.log('Using sub dataset that only consider images containing objects')
        measures2d, measure_mean2d, measure_std2d = evaluer.eval(i_labels=self.labels, i_preds=self.preds, i_object_care=True)
        Logs.log('Measure shape (2D) = {}'.format(measures2d.shape))
        Logs.log('Measure mean  (2D) = {}'.format(measure_mean2d))
        Logs.log('Measure std   (2D) = {}'.format(measure_std2d))
        self.erase_indicators()
        return measures,SegMetrics.get_mean(measures),SegMetrics.get_std(measures),global_measures
"""=================================================================================================================="""
if __name__ == '__main__':
    print('This module is to implement the metrics for evaluating performance of segmentation networks')
    labels = np.load(r'F:\officials\sources\py\Projects\BrainStrokes\ckpts\Fold_1_of_5\val_labels.npy')
    preds  = np.load(r'F:\officials\sources\py\Projects\BrainStrokes\ckpts\Fold_1_of_5\val_preds.npy')
    labels = np.squeeze(labels)
    preds  = np.squeeze(preds)
    labels = (labels>0).astype(np.int)
    preds  = (preds>0).astype(np.int)
    print(labels.shape,preds.shape)
    exampler = SegMetrics(i_num_classes=2,i_care_background=False)
    ex_measures,ex_measure_mean,ex_measure_std = exampler.eval(i_labels=labels,i_preds=preds,i_object_care=False)
    print(ex_measures.shape)
    print(ex_measure_mean)
    print(ex_measure_std)
    ex_measures, ex_measure_mean, ex_measure_std = exampler.eval(i_labels=labels, i_preds=preds, i_object_care=True)
    print(ex_measures.shape)
    print(ex_measure_mean)
    print(ex_measure_std)
"""=================================================================================================================="""