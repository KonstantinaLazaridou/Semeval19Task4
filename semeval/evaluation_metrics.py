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
    tp = 0;
    fp = 0;
    fn = 0;
    tn = 0;
    for index, row in result.iterrows():
        if row["predicted"] == row["actual"]:
            if row["predicted"] == 1 and row["actual"] == 1:
                tp += 1;
            elif row["predicted"] == 0 and row["actual"] == 0:
                tn += 1;
        else:
            if row["predicted"] == 1 and row["actual"] == 0:
                fp += 1;
            elif row["predicted"] == 0 and row["actual"] == 1:
                fn += 1;
    print("tp = {}, fp = {}, fn = {}, tn = {}".format(tp, fp, fn, tn));