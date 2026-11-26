from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay

def evaluate(y_true, y_pred):
    p,r,f,_ = precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    return dict(accuracy=accuracy_score(y_true,y_pred),precision=p,recall=r,f1=f)

def save_confusion_matrix(y_true,y_pred,path):
    import matplotlib.pyplot as plt
    ConfusionMatrixDisplay.from_predictions(y_true,y_pred,xticks_rotation=45)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
