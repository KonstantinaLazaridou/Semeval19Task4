from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def compute_confusion_matrix(result, pos_label):
    targets = ([str(label).replace("\n", "") for label in result["actual"].values]);
    predictions = ([str(label).replace("\n", "") for label in result["predicted"].values]);
    print("Evaluating...");
    print("Targets: {}".format(targets));
    print("Predictions: {}".format(predictions));
    pr = precision_score(y_true=targets, y_pred=predictions, pos_label=pos_label);
    rec = recall_score(y_true=targets, y_pred=predictions, pos_label=pos_label);
    f1 = f1_score(y_true=targets, y_pred=predictions, pos_label=pos_label);
    acc = accuracy_score(y_true=targets, y_pred=predictions);
    print("Precision={}, Recall={}, F1={}, Accuracy={}".format(pr, rec, f1, acc));
    print("{0:.2f}\t{1:.2f}\t{2:.2f}\t{3:.2f}\n".format(pr, rec, f1, acc));