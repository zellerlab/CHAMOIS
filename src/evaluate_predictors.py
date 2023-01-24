import itertools
import json
import gzip
import csv
import pickle
import os

import anndata
import disjoint_set
import fisher
import numpy
import pronto
import scipy.sparse
import rich.progress
import rich.table
import rich.panel
import sklearn.tree
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neighbors
import matplotlib
import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Bold_10

matplotlib.use('Agg')


# --- sklearn Feature Selection implementation -------------------------------

class SelectPValueUnderThreshold(sklearn.base.TransformerMixin):
    """Select features with Fisher p-value under a certain threshold.
    """

    def __init__(self, threshold=1.0):
        self.threshold = threshold
        self.features_ = None
        self.pvalues_ = None

    def fit(self, X, y):
        self.pvalues_ = numpy.zeros(X.shape[1], dtype=numpy.float32)
        for feature in range(X.shape[1]):
            x = X[:,feature]       
            result = fisher.pvalue(
                ((x > 0) & (y == 1)).sum(),
                ((x == 0) & (y == 1)).sum(),
                ((x > 0) & (y == 0)).sum(),
                ((x == 0) & (y == 0)).sum(),
            )
            self.pvalues_[feature] = result.two_tail
        self.indices_ = numpy.where(self.pvalues_ < self.threshold)[0]
        return self

    def transform(self, X):
        if self.indices_ is None:
            raise sklearn.exceptions.NotFittedError("model was not fitted")
        if X.shape[1] != self.pvalues_.shape[0]:
            raise ValueError(f"X has {X.shape[1]} features, but SelectPValueUnderThreshold is expecting {self.pvalues_.shape[0]} features as input.")
        return X[:, self.indices_]

# --- Load features and classes ----------------------------------------------

features = anndata.read("data/datasets/mibig3.1/pfam35.hdf5")
classes = anndata.read("data/datasets/mibig3.1/classes.hdf5")
assert (features.obs.index == classes.obs.index).all()


# --- Load ChemOnt -----------------------------------------------------------

chemont = pronto.Ontology("data/chemont/ChemOnt_2_1.obo")


# --- Build one classifier per class -----------------------------------------

CV_FOLDS = 5
CLASSIFIERS = {
    "LogisticRegression(penalty='l1')": sklearn.linear_model.LogisticRegression(penalty="l1", solver="liblinear"),
    "RandomForest()": sklearn.ensemble.RandomForestClassifier(),
}

os.makedirs(os.path.join("build", "classifier"), exist_ok=True)

with rich.progress.Progress() as progress:

    # record curves and stats even for partially interrupted training
    curves = []
    rows = []
    ginis = numpy.zeros((classes.shape[1], features.shape[1]))

    try:
        # only use BGCs with known compound structure and class assignment
        features = features[~classes.obs.unknown_structure]
        classes = classes[~classes.obs.unknown_structure]
        rich.print(f"Using {features.n_vars} BGCs with compound classification")

        # TODO: better stratification to get rid of possible duplicate BGCs, use ANI matrix
        for class_idx, class_name in enumerate(progress.track(classes.var_names, description=f"[bold blue]{'Processing':>12}[/]")):
            # use labels for current class
            y_true = classes.X.toarray()[:, class_idx]

            # skip classes with no negative or no positives
            if y_true.sum() < CV_FOLDS or y_true.sum() >= len(y_true) - CV_FOLDS:
                continue
            elif numpy.unique(classes.obs.groups[y_true]).shape[0] < CV_FOLDS:
                continue
            else:
                rich.print(f"Training [bold blue]{class_name}[/] ({chemont[class_name].name!r}) classifier with {y_true.sum()} positive and {(~y_true).sum()} negatives")

            # show example BGCs belonging to positive or negative
            #neg = next(mibig[names_masked[i]] for i in range(y_true.shape[0]) if not y_true[i])
            #pos = next(mibig[names_masked[i]] for i in range(y_true.shape[0]) if y_true[i])
            #rich.print(f"Examples: positive {pos['mibig_accession']} ({pos['compounds'][0]['compound']!r}) / negative {neg['mibig_accession']} ({neg['compounds'][0]['compound']!r})")

            auprs = {}
            aurocs = {}
            losses = {}
            curve_index = len(curves)

            # train all classifiers
            task = progress.add_task(total=len(CLASSIFIERS), description=f"[bold blue]{'Training':>12}[/]")
            for classifier_name, classifier in progress.track(CLASSIFIERS.items(), task_id=task):
                # cross-validate predictor in cross-validation and compute stats
                try:
                    y_pred = sklearn.model_selection.cross_val_predict(
                        classifier, 
                        features.X, 
                        y_true, 
                        method="predict_proba", 
                        cv=sklearn.model_selection.GroupKFold(CV_FOLDS), 
                        n_jobs=-1,
                        groups=classes.obs.groups,
                    )[:, 1]
                except AttributeError:
                    y_pred = sklearn.model_selection.cross_val_predict(
                        classifier, 
                        features.X, 
                        y_true, 
                        method="predict", 
                        cv=sklearn.model_selection.GroupKFold(CV_FOLDS), 
                        n_jobs=-1,
                        groups=classes.obs.groups,
                    )

                # compute evaluation statistics
                fp, tp, _ = sklearn.metrics.roc_curve(y_true, y_pred)
                aurocs[classifier_name] = sklearn.metrics.roc_auc_score(y_true, y_pred)
                precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred)
                auprs[classifier_name] = sklearn.metrics.auc(recall, precision)
                losses[classifier_name] = sklearn.metrics.log_loss(y_true, y_pred)
                # rich.print(f"Results for [bold blue]{class_name}[/] ({chemont[class_name].name!r}) with {classifier_name}: AUROC={aurocs[classifier_name]:.3f} AUPR={auprs[classifier_name]:.3f} loss={losses[classifier_name]:.3f}")
                curves.append({"class": class_name, "kind": "precision-recall", "x": list(recall), "y": list(precision), "classifier": classifier_name})
                curves.append({"class": class_name, "kind": "roc", "x": list(fp), "y": list(tp), "classifier": classifier_name})

                # save statistics
                rows.append([
                    class_name,
                    chemont[class_name].name,
                    y_true.sum(),
                    (~y_true).sum(),
                    classifier_name,
                    aurocs[classifier_name],
                    auprs[classifier_name],
                    losses[classifier_name],
                ])
            progress.update(task_id=task, visible=False)

            # show table of statistics
            table = rich.table.Table("classifier", "auroc", "aupr", "loss")
            for classifier_name in CLASSIFIERS:
                aupr, auroc, loss = auprs[classifier_name], aurocs[classifier_name], losses[classifier_name]
                table.add_row(
                    classifier_name, 
                    rich.text.Text(format(auroc, ".3f"), style="bold green" if auroc == max(aurocs.values()) else ""),
                    rich.text.Text(format(aupr, ".3f"), style="bold green" if aupr == max(auprs.values()) else ""),
                    rich.text.Text(format(loss, ".3f"), style="bold green" if loss == min(losses.values()) else ""),
                )
            rich.print(table)

            # get most contributing domains with Gini index and p-values
            rf = sklearn.ensemble.RandomForestClassifier().fit(features.X, y_true)
            ginis[class_idx, :] = rf.feature_importances_

            # plot precision/recall and ROC curves
            palette = {
                "LogisticRegression(penalty='l1')": Bold_10.hex_colors[0],
                "LogisticRegression(penalty='l1')+Selection": Bold_10.hex_colors[0],
                "DecisionTreeClassifier()": Bold_10.hex_colors[2],
                "DecisionTreeClassifier()+Selection": Bold_10.hex_colors[2],
                "RandomForest(10)": Bold_10.hex_colors[1],
                "RandomForest(10)+Selection": Bold_10.hex_colors[1],
                "RandomForest()": Bold_10.hex_colors[5],
                "RandomForest()+Selection": Bold_10.hex_colors[5],
                "MultinomialNB()": Bold_10.hex_colors[4],
                "MultinomialNB()+Selection": Bold_10.hex_colors[4],
                "KNeighbours(metric='cityblock')": Bold_10.hex_colors[6],
                "KNeighbours(metric='cityblock')+Selection": Bold_10.hex_colors[6],
                "RandomForest(max_depth=10)": Bold_10.hex_colors[7],
                "RandomForest(max_depth=10)+Selection": Bold_10.hex_colors[7],
                "AdaBoostClassifier()": Bold_10.hex_colors[8],
                "AdaBoostClassifier()+Selection": Bold_10.hex_colors[8],
            }
            styles = {
                "LogisticRegression(penalty='l1')": "-",
                "LogisticRegression(penalty='l1')+Selection": "--",
                "DecisionTreeClassifier()": "-",
                "DecisionTreeClassifier()+Selection": "--",
                "RandomForest(10)": "-",
                "RandomForest(10)+Selection": "--",
                "RandomForest()": "-",
                "RandomForest()+Selection": "--",
                "MultinomialNB()": "-",
                "MultinomialNB()+Selection": "--",
                "KNeighbours(metric='cityblock')": "-",
                "KNeighbours(metric='cityblock')+Selection": "--",
                "RandomForest(max_depth=10)": "-",
                "RandomForest(max_depth=10)+Selection": "--",
                "AdaBoostClassifier()": "-",
                "AdaBoostClassifier()+Selection": "--",
            }

            plt.figure(1, figsize=(8, 8))
            plt.clf()
            for curve in curves[curve_index+1::2]:
                auc = sklearn.metrics.auc(curve["x"], curve["y"])
                plt.plot(curve["x"], curve["y"], label=f"{curve['classifier']} ({auc:.3})", color=palette[curve['classifier']], linestyle=styles[curve['classifier']])
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
            plt.title(f"{class_name} ({chemont[class_name].name}): {y_true.sum()} members ")
            plt.savefig(f"build/classifier/{class_name}.roc.png")

            plt.figure(2, figsize=(8, 8))
            plt.clf()
            for curve in curves[curve_index::2]:
                auc = sklearn.metrics.auc(curve["x"], curve["y"])
                plt.plot(curve["x"], curve["y"], label=f"{curve['classifier']} ({auc:.3})", color=palette[curve['classifier']], linestyle=styles[curve['classifier']])
            plt.legend()
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.axhline(y_true.sum() / y_true.shape[0], linestyle="--", color="grey")
            plt.title(f"{class_name} ({chemont[class_name].name}): {y_true.sum()} members")
            plt.savefig(f"build/classifier/{class_name}.pr.png")

    finally:
        # save stats
        with open("build/classifier-stats.tsv", "w") as dst:
            writer = csv.writer(dst, dialect="excel-tab")
            writer.writerow(["class", "name", "positives", "negatives", "classifier", "auroc", "aupr", "loss"])
            writer.writerows(rows)
        # save curves
        with open("build/classifier-curves.json", "w") as dst:
            json.dump(curves, dst)
        # save GINI indices
        # with open("build/classifier-gini.npz", "wb") as dst:
        #     numpy.savez_compressed(dst, domains=domains, classes=numpy.array(list(class_index)), gini=ginis)

