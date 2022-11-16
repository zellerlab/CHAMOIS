import itertools
import json
import gzip
import csv

import fisher
import numpy
import pronto
import scipy.sparse
import rich.progress
import sklearn.ensemble
import sklearn.model_selection
import pyhmmer

# --- Load Pfam domain names -------------------------------------------------

with rich.progress.open("data/Pfam35.0.hmm.gz", "rb") as src:
    with pyhmmer.plan7.HMMFile(gzip.open(src)) as hmm_file:
        pfam_names = {
            hmm.accession.decode(): hmm.name.decode()
            for hmm in hmm_file
        }

# --- Load MIBiG -------------------------------------------------------------

with open("build/mibig-classified.json", "rb") as src:
    mibig = json.load(src)


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
labels = numpy.zeros((len(names), len(class_index)), dtype=numpy.bool_)

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

# record curves and stats even for partially interrupted training
curves = []
rows = []


try:
    # only use BGCs with known compound structure and class assignment
    names_masked = names[~mask]
    labels_masked = labels[~mask, :]
    features_masked = features[~mask, :]

    # TODO: better stratification to get rid of possible duplicate BGCs
    for class_name in rich.progress.track(class_index, description="Training..."):

        # use a random forest classifier
        rf = sklearn.ensemble.RandomForestClassifier()
        y_true = labels_masked[:, class_index[class_name]]

        # skip classes with no negative or no positives
        if y_true.sum() <= 1 or y_true.sum() >= len(y_true) - 1:
            rich.print(f"Skipping [bold blue]{class_name}[/] ({chemont[class_name].name!r}) without class members")
            rich.print()
            continue
        else:
            rich.print(f"Training [bold blue]{class_name}[/] ({chemont[class_name].name!r}) classifier with {y_true.sum()} positive and {(~y_true).sum()} negatives")

        # show example BGCs belonging to positive or negative
        neg = next(mibig[names_masked[i]] for i in range(y_true.shape[0]) if not y_true[i])
        pos = next(mibig[names_masked[i]] for i in range(y_true.shape[0]) if y_true[i])
        rich.print(f"Example positive: {pos['mibig_accession']} ({pos['compounds'][0]['compound']!r})")
        rich.print(f"Example negative: {neg['mibig_accession']} ({neg['compounds'][0]['compound']!r})")

        # cross-validate predictor in cross-validation and compute stats
        y_pred = sklearn.model_selection.cross_val_predict(rf, features_masked, y_true, method="predict_proba", cv=min(5, y_true.sum()), n_jobs=-1)[:, 1]
        fp, tp, _ = sklearn.metrics.roc_curve(y_true, y_pred)
        auroc_all = sklearn.metrics.roc_auc_score(y_true, y_pred)
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred)
        aupr_all = sklearn.metrics.auc(recall, precision)
        loss_all = sklearn.metrics.log_loss(y_true, y_pred)
        rich.print(f"Results for [bold blue]{class_name}[/] ({chemont[class_name].name!r}) with whole Pfam: AUROC={auroc_all:.3f} AUPR={aupr_all:.3f} loss={loss_all:.3f}")
        curves.append({"class": class_name, "features": "full", "kind": "precision-recall", "x": list(recall), "y": list(precision)})
        curves.append({"class": class_name, "features": "full", "kind": "roc", "x": list(fp), "y": list(tp)})

        # cross-validate predictor with selected features
        pvalues = compute_pvalues(features_masked, y_true)
        domains_selected = domains[pvalues < 0.1]
        features_selected = features_masked[:, pvalues < 0.1]
        y_pred = sklearn.model_selection.cross_val_predict(rf, features_selected, y_true, method="predict_proba", cv=min(5, y_true.sum()), n_jobs=-1)[:, 1]
        fp, tp, _ = sklearn.metrics.roc_curve(y_true, y_pred)
        auroc_selected = sklearn.metrics.roc_auc_score( y_true, y_pred )
        precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred)
        aupr_selected = sklearn.metrics.auc(recall, precision)
        loss_selected = sklearn.metrics.log_loss(y_true, y_pred)
        rich.print(f"Results for [bold blue]{class_name}[/] ({chemont[class_name].name!r}) with {features_selected.shape[1]} Pfam domains: AUROC={auroc_selected:.3f} AUPR={aupr_selected:.3f} loss={loss_selected:.3f}")
        curves.append({"class": class_name, "features": "selected", "kind": "precision-recall", "x": list(recall), "y": list(precision)})
        curves.append({"class": class_name, "features": "selected", "kind": "roc", "x": list(fp), "y": list(tp)})

        # get most contributing domains with Gini index and p-values
        rf.fit(features_masked, y_true)
        highest_contributors = domains[ numpy.argsort(rf.feature_importances_)[-5:] ]
        rich.print(f"Highest contributors:")
        for accession in highest_contributors:
            domain_index = numpy.where(domains == accession)[0][0]
            rich.print(f"- {accession} ({pfam_names[accession]!r}) p={pvalues[domain_index]} Gini={rf.feature_importances_[domain_index]}")

        # save statistics
        rich.print()
        rows.append([
            class_name,
            chemont[class_name].name,
            y_true.sum(),
            (~y_true).sum(),
            features_selected.shape[1],
            auroc_all,
            aupr_all,
            auroc_selected,
            aupr_selected,
            ";".join(highest_contributors)
        ])

finally:
    # save stats
    with open("build/classifier-stats.tsv", "w") as dst:
        writer = csv.writer(dst, dialect="excel-tab")
        writer.writerow(["class", "name", "positives", "negatives", "selected_features", "auroc_all", "aupr_all", "auroc_selected", "aupr_selected", "highest_contributors"])
        writer.writerows(rows)
    # save curves
    with open("build/classifier-curves.json", "w") as dst:
        json.dump(curves, dst)





# # --- Assign ChemOnt annotations to MIBiG clusters ---------------------------

# mibig_superclass = {
#     mibig_id: compound['superclass']['chemont_id']
#     for mibig_id, compounds in mibig_classyfire.items()
#     for compound in compounds
#     if compound['superclass'] is not None and compound['superclass']['chemont_id'] is not None
# }

# mibig_class = {
#     mibig_id: compound['class']['chemont_id']
#     for mibig_id, compounds in mibig_classyfire.items()
#     for compound in compounds
#     if compound['class'] is not None and compound['class']['chemont_id'] is not None
# }
    
# mibig_parent = {
#     mibig_id: compound['direct_parent']['chemont_id']
#     for mibig_id, compounds in mibig_classyfire.items()
#     for compound in compounds
#     if compound['direct_parent'] is not None and compound['direct_parent']['chemont_id'] is not None
# }

# for mibig_id, compounds in mibig_classyfire.items():
#     for compound in compounds:
#         if compound is not None and "superclass" not in compound:
#             print(compound.keys())
#             break

# counts = collections.Counter([ mibig_superclass[bgc] for bgc in bgcs if bgc in mibig_superclass ])
# counts.most_common()        