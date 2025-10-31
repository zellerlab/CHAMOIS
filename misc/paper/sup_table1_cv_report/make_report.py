import pathlib
import pandas

folder = pathlib.Path(__file__).parent

# PROJECT_FOLDER = folder
# while not PROJECT_FOLDER.joinpath("chamois").exists():
#     PROJECT_FOLDER = PROJECT_FOLDER.parent

rf = pandas.read_table(folder.parent.joinpath("fig2_cross_validation", "rf.report.tsv"))
rf = rf[['class', 'auroc', 'auprc', 'f1_score']]
rf.columns = ['class_accession', 'rf_auroc', 'rf_auprc', 'rf_f1_score']

cv = pandas.read_table(folder.parent.joinpath("fig2_cross_validation", "cv.report.tsv"))
cv = cv[['name', 'description', 'n_positives', 'class', 'auprc', 'auroc', 'f1_score']]
cv.columns = ['class_name', 'class_description', 'n_positives', 'class_accession', 'logistic_auprc', 'logistic_auroc', 'logistic_f1_score']

out = pandas.merge(cv, rf, on="class_accession")
out.to_csv(folder.joinpath('report.tsv'), sep='\t', index=False)
