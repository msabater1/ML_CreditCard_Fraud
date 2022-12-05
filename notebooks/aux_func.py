from sklearn.metrics import r2_score, precision_recall_curve, recall_score, precision_score, auc, classification_report, fbeta_score, ConfusionMatrixDisplay, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as ss

# Métricas: F2, R2, Accuracy, Precision, Recall, F1, Support, Matriz de Confusion
def plot_metrics(yval, y_pred, yhat=None, Y=None):

    try:
        if yhat.all() != None:
            # F2-Score
            print(f'\nF2 Score: {fbeta_score(yval, y_pred, beta=2, average="macro")}\n') 
            # R2
            print(f'R2 Score: {r2_score(yval, y_pred)}\n') 
            # Accuracy
            print(f'Accuracy Score: {accuracy_score(yval, y_pred)}\n') 
            # Tabla con precision, recall, f1-score, support
            print(classification_report(yval, y_pred, labels= [0, 1])) 

            #Confusion Matrix
            disp = ConfusionMatrixDisplay.from_predictions(yval, y_pred, cmap=plt.cm.Blues)
            disp.figure_.suptitle("Confusion Matrix")
            # Confusion Matrix Normalized
            disp = ConfusionMatrixDisplay.from_predictions(yval, y_pred, cmap=plt.cm.Blues,
                                                               normalize ='true')
            disp.figure_.suptitle("Confusion Matrix Normalize")

            # retrieve just the probabilities for the positive class
            pos_probs = yhat[:, 1]
            # calculate the no skill line as the proportion of the positive class
            no_skill = len(Y[Y==1]) / len(Y)
            # plot the no skill precision-recall curve
            fig = plt.figure()
            ax = plt.axes()
            ax.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
            # calculate model precision-recall curve
            precision, recall, thresholds = precision_recall_curve(yval, pos_probs)
            # convert to f score
            fscore = (2 * precision * recall) / (precision + recall)
            # locate the index of the largest f score
            ix = np.argmax(fscore)
            # plot the model precision-recall curve
            ax.plot(recall, precision, marker='.', label='Model')
            # Best Threshold
            ax.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')
            # axis labels
            plt.xlabel('Recall')
            plt.ylabel('Precision') 
            plt.title('Precision-Recall Curve')
            # show the legend
            plt.legend()
            # show the plot
            plt.show()

            print(f'Model PR AUC: {auc(recall, precision)}')
            print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
            
    except AttributeError:
            print(f'\nF2 Score: {fbeta_score(yval, y_pred, beta=2, average="macro")}\n') # F2-Score

            print(f'R2 Score: {r2_score(yval, y_pred)}\n') # R2

            print(f'Accuracy Score: {accuracy_score(yval, y_pred)}\n') # Accuracy

            print(classification_report(yval, y_pred, labels= [0, 1])) # Tabla con precision, recall, f1-score, support

            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true', values_format='.2g') # Confusion Matrix
            disp.figure_.suptitle("Percentage Confusion Matrix")

            disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred) # Confusion Matrix
            disp.figure_.suptitle("Confusion Matrix")
            # print(f"Confusion matrix:\n{disp.confusion_matrix}") # devuelve la matrix en texto
            
def plot_recall_precission(recall_precision):

    plt.figure(figsize=(15, 5))
    ax = sns.pointplot(x = [element[0] for element in recall_precision], y=[element[1] for element in recall_precision],
                     color="r", label='recall', scale=1)
    ax = sns.pointplot(x = [element[0] for element in recall_precision], y=[element[2] for element in recall_precision],
                     color="b", label='precission')
    ax.set_title('recall-precision versus threshold')
    ax.set_xlabel('threshold')
    ax.set_ylabel('probability')
    ax.legend()

    labels = ax.get_xticklabels()
    for i,l in enumerate(labels):
        if(i%5 == 0) or (i%5 ==1) or (i%5 == 2) or (i%5 == 3):
            labels[i] = '' # skip even labels
            ax.set_xticklabels(labels, rotation=45, fontdict={'size': 10})
    plt.show()

def plot_threshold(model, x_train_scaled, x_test_scaled, y_train, y_test):
        # predicción de TRAIN
    pd_train_predicted = pd.DataFrame(model.predict_proba(x_train_scaled), 
                                    index=x_train_scaled.index, columns = ['y_predicted_0', 'y_predicted']).drop(['y_predicted_0'],axis=1)
    pd_train_predicted_final = pd.concat([x_train_scaled, pd_train_predicted, y_train],axis=1)

    prob_predictions = pd_train_predicted_final.y_predicted.values
    recall_precision = []

    for threshold in np.arange(0.01, 0.99, 0.01):
        
        given_threshold = [1 if value>threshold else 0 for value in prob_predictions]
        recall_precision.append([threshold, recall_score(pd_train_predicted_final.isFraud, given_threshold),
                                precision_score(pd_train_predicted_final.isFraud, given_threshold)])

    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plot_recall_precission(recall_precision)