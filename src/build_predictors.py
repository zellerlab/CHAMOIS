import itertools
import json
import gzip
import csv
import pickle
import os

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
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neighbors
import matplotlib.pyplot as plt
from palettable.cartocolors.qualitative import Bold_9


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


# --- Constants --------------------------------------------------------------

CV_FOLDS = 5


# --- Load Pfam domain names -------------------------------------------------

with open("data/Pfam35.0.txt") as src:
    pfam_names = dict(line.strip().split("\t", 1) for line in src)


# --- Load MIBiG -------------------------------------------------------------

with open("build/mibig-classified.json", "rb") as src:
    mibig = json.load(src)


# --- Load ANI matrix and build groups based on ANI --------------------------

with open("data/mibig/ani_3.1.coo.npz", "rb") as src:
    animatrix = scipy.sparse.load_npz(src).todok()
    group_set = disjoint_set.DisjointSet({ i:i for i in range(len(mibig)) })
    indices = itertools.combinations(range(len(mibig)), 2)
    total = len(mibig) * (len(mibig) - 1) / 2
    for (i, j) in rich.progress.track(indices, total=total, description="Grouping..."):
        if animatrix[i, j] >= 0.8:
            group_set.union(i, j)
    groups = numpy.array([group_set[i] for i in range(len(mibig))])


# --- Load MIBiG compositions ------------------------------------------------

with open("build/compositions/Pfam35.0/counts.npz", "rb") as comp_src:
    counts = scipy.sparse.load_npz(comp_src).toarray()
with open("build/compositions/Pfam35.0/domains.tsv") as domains_src:
    domains = numpy.array([ line.split("\t")[0].strip() for line in domains_src ])
with open("build/compositions/Pfam35.0/labels.tsv") as typs_src:
    names = numpy.array([ line.split("\t")[0].strip() for line in typs_src ])

# remove empty features
domains = domains[counts.sum(axis=0) > 0]
features = counts[:,counts.sum(axis=0) > 0]
rich.print(f"Using {features.shape[1]} non-null features")


# --- Load ChemOnt -----------------------------------------------------------

chemont = pronto.Ontology("data/chemont/ChemOnt_2_1.obo")
rich.print(f"Loaded {len(chemont)} terms from ChemOnt ontology")

class_index = { 
    term.id: i
    for i, term in enumerate(chemont.terms())
    if term.id != "CHEMONTID:9999999"
}


# --- Get classes ------------------------------------------------------------

mask = numpy.zeros(len(names), dtype=numpy.bool_)
labels = numpy.zeros((len(names), len(class_index)+1), dtype=numpy.bool_)

for bgc_id, bgc in rich.progress.track(mibig.items(), description="Preparing labels..."):
    # skip and ignore BGCs without any classyfire annotation
    if not any("classyfire" in compound for compound in bgc["compounds"]):
        mask[ numpy.where(names == bgc_id) ] = True
        continue
    # go through all BGC compounds
    for compound in bgc["compounds"]:
        # skip compounds without classifyre annotations
        if compound.get("classyfire") is None:
            continue
        # get all parents by traversing the ontology transitively
        direct_parents = pronto.TermSet({
            chemont[direct_parent["chemont_id"]] # type: ignore
            for direct_parent in itertools.chain(
                [compound["classyfire"]["kingdom"]],  
                [compound["classyfire"]["superclass"]],  
                [compound["classyfire"]["class"]],  
                [compound["classyfire"]["subclass"]],  
                [compound["classyfire"]["direct_parent"]],  
                compound["classyfire"]["intermediate_nodes"],
                compound["classyfire"]["alternative_parents"],
            )
            if direct_parent is not None
        })
        all_parents = direct_parents.superclasses().to_set()
        # set label flag is compound belong to a class
        bgc_index = numpy.where(names == bgc_id)
        for parent in all_parents:
            if parent.id != "CHEMONTID:9999999":
                labels[bgc_index, class_index[parent.id]] = True

# --- Use Fisher's exact test to select features -----------------------------

def compute_pvalues(X, y, kind="two_tail"):
    pvalues = []
    for feature in range(X.shape[1]):
        x = X[:,feature]       
        result = fisher.pvalue(
            ((x > 0) & (y == 1)).sum(),
            ((x == 0) & (y == 1)).sum(),
            ((x > 0) & (y == 0)).sum(),
            ((x == 0) & (y == 0)).sum(),
        )
        pvalues.append(getattr(result, kind))
    return numpy.array(pvalues)


# --- Build one classifier per class -----------------------------------------

CLASSIFIERS = {
    # a simple Lasso to see the individual weights
    "Lasso()": sklearn.linear_model.Lasso(),
    # random forest, either 1, 10 or 100 trees
    "DecisionTreeClassifier()": sklearn.tree.DecisionTreeClassifier(),
    "RandomForest(10)": sklearn.ensemble.RandomForestClassifier(10),
    "RandomForest(100)": sklearn.ensemble.RandomForestClassifier(100),
    "RandomForest(max_depth=6)": sklearn.ensemble.RandomForestClassifier(max_depth=6),
    # multinomialNB
    "MultinomialNB()": sklearn.naive_bayes.MultinomialNB(),
    # KNN with cityblock metric (addding/substracting domains)
    "KNeighbours(metric='cityblock')": sklearn.neighbors.KNeighborsClassifier(metric="cityblock"),
    # variants with feature selection
    "DecisionTreeClassifier()+Selection": sklearn.pipeline.Pipeline([
        ("feature_selection", SelectPValueUnderThreshold(0.1)),
        ("classifier", sklearn.tree.DecisionTreeClassifier()),
    ]),
    "RandomForest(10)+Selection": sklearn.pipeline.Pipeline([
        ("feature_selection", SelectPValueUnderThreshold(0.1)),
        ("classifier", sklearn.ensemble.RandomForestClassifier(10))
    ]),
    "RandomForest(100)+Selection": sklearn.pipeline.Pipeline([
        ("feature_selection", SelectPValueUnderThreshold(0.1)),
        ("classifier", sklearn.ensemble.RandomForestClassifier(100))
    ]),
    "RandomForest(max_depth=6)+Selection": sklearn.pipeline.Pipeline([
        ("feature_selection", SelectPValueUnderThreshold(0.1)),
        ("classifier", sklearn.ensemble.RandomForestClassifier(max_depth=6),)
    ]),
    "MultinomialNB()+Selection": sklearn.pipeline.Pipeline([
        ("feature_selection", SelectPValueUnderThreshold(0.1)),
        ("classifier", sklearn.naive_bayes.MultinomialNB())
    ]),
    "KNeighbours(metric='cityblock')+Selection": sklearn.pipeline.Pipeline([
        ("feature_selection", SelectPValueUnderThreshold(0.1)),
        ("classifier", sklearn.neighbors.KNeighborsClassifier(metric="cityblock"))
    ])
}

with rich.progress.Progress() as progress:

    # record curves and stats even for partially interrupted training
    curves = []
    rows = []
    ginis = numpy.zeros((len(class_index), features.shape[1]))

    try:
        # only use BGCs with known compound structure and class assignment
        names_masked = names[~mask]
        labels_masked = labels[~mask, :]
        groups_masked = groups[~mask]
        features_masked = features[~mask, :]
        rich.print(f"Using {features_masked.shape[0]} BGCs with compound classification")

        # TODO: better stratification to get rid of possible duplicate BGCs, use ANI matrix
        for class_name in progress.track(class_index, description="Processing..."):
            # use labels for current class
            y_true = labels_masked[:, class_index[class_name]]

            # skip classes with no negative or no positives
            if y_true.sum() < CV_FOLDS or y_true.sum() >= len(y_true) - CV_FOLDS:
                continue
            else:
                rich.print(f"Training [bold blue]{class_name}[/] ({chemont[class_name].name!r}) classifier with {y_true.sum()} positive and {(~y_true).sum()} negatives")

            # show example BGCs belonging to positive or negative
            neg = next(mibig[names_masked[i]] for i in range(y_true.shape[0]) if not y_true[i])
            pos = next(mibig[names_masked[i]] for i in range(y_true.shape[0]) if y_true[i])
            rich.print(f"Examples: positive {pos['mibig_accession']} ({pos['compounds'][0]['compound']!r}) / negative {neg['mibig_accession']} ({neg['compounds'][0]['compound']!r})")

            auprs = {}
            aurocs = {}
            losses = {}
            curve_index = len(curves)

            # train all classifiers
            task = progress.add_task(total=len(CLASSIFIERS), description="Training...")
            for classifier_name, classifier in progress.track(CLASSIFIERS.items(), task_id=task):
                # cross-validate predictor in cross-validation and compute stats
                try:
                    y_pred = sklearn.model_selection.cross_val_predict(
                        classifier, 
                        features_masked, 
                        y_true, 
                        method="predict_proba", 
                        cv=sklearn.model_selection.GroupKFold(CV_FOLDS), 
                        n_jobs=-1,
                        groups=groups_masked,
                    )[:, 1]
                except AttributeError:
                    y_pred = sklearn.model_selection.cross_val_predict(
                        classifier, 
                        features_masked, 
                        y_true, 
                        method="predict", 
                        cv=sklearn.model_selection.GroupKFold(CV_FOLDS), 
                        n_jobs=-1,
                        groups=groups_masked,
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
            rf = sklearn.ensemble.RandomForestClassifier().fit(features_masked, y_true)
            ginis[class_index[class_name], :] = rf.feature_importances_

            # plot precision/recall and ROC curves
            palette = {
                "Lasso()": Bold_9.hex_colors[0],
                "DecisionTreeClassifier()": Bold_9.hex_colors[2],
                "DecisionTreeClassifier()+Selection": Bold_9.hex_colors[2],
                "RandomForest(10)": Bold_9.hex_colors[1],
                "RandomForest(10)+Selection": Bold_9.hex_colors[1],
                "RandomForest(100)": Bold_9.hex_colors[5],
                "RandomForest(100)+Selection": Bold_9.hex_colors[5],
                "MultinomialNB()": Bold_9.hex_colors[4],
                "MultinomialNB()+Selection": Bold_9.hex_colors[4],
                "KNeighbours(metric='cityblock')": Bold_9.hex_colors[6],
                "KNeighbours(metric='cityblock')+Selection": Bold_9.hex_colors[6],
                "RandomForest(max_depth=6)": Bold_9.hex_colors[7],
                "RandomForest(max_depth=6)+Selection": Bold_9.hex_colors[7],
            }
            styles = {
                "Lasso()": "-",
                "DecisionTreeClassifier()": "-",
                "DecisionTreeClassifier()+Selection": "--",
                "RandomForest(10)": "-",
                "RandomForest(10)+Selection": "--",
                "RandomForest(100)": "-",
                "RandomForest(100)+Selection": "--",
                "MultinomialNB()": "-",
                "MultinomialNB()+Selection": "--",
                "KNeighbours(metric='cityblock')": "-",
                "KNeighbours(metric='cityblock')+Selection": "--",
                "RandomForest(max_depth=6)": "-",
                "RandomForest(max_depth=6)+Selection": "--",
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
        with open("build/classifier-gini.npz", "wb") as dst:
            numpy.savez_compressed(dst, domains=domains, classes=numpy.array(list(class_index)), gini=ginis)

