
# Third-party modules
import sklearn.metrics as sk
import matplotlib.pyplot as plt

class Performance:

    def __init__(self, true, pred):
        self.true = true
        self.pred = pred

    @property
    def accuracy(self):
        return sk.accuracy_score(self.true, self.pred)
    
    @property
    def balanced_accuracy(self):
        return sk.balanced_accuracy_score(self.true, self.pred)
    
    @property
    def precision(self):
        return sk.precision_score(self.true, self.pred)
    
    @property
    def recall(self):
        return sk.recall_score(self.true, self.pred)
    
    @property
    def f1(self):
        return sk.f1_score(self.true, self.pred)
    
    @property
    def mcc(self):
        return sk.matthews_corrcoef(self.true, self.pred)
    
    @property
    def confusion_matrix(self):
        return sk.confusion_matrix(self.true, self.pred)
    
    def plot_confusion_matrix(self):
        cm = sk.confusion_matrix(self.true, self.pred)
        cm = sk.ConfusionMatrixDisplay(cm, display_labels = ['-', '+'])
        cm.plot()
        plt.savefig('confusion_matrix.png')
        plt.clf()
    
    @property
    def classification_report(self):
        return sk.classification_report(self.true, self.pred)

if __name__ == '__main__':
    true = [1,1,0,0,1,0,1,0,1,0]
    pred = [1,1,0,0,1,0,1,0,0,0]
    performance = Performance(true, pred)
    performance.plot_confusion_matrix()