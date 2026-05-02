import pathlib
import pandas

folder = pathlib.Path(__file__).parent

# PROJECT_FOLDER = folder
# while not PROJECT_FOLDER.joinpath("chamois").exists():
#     PROJECT_FOLDER = PROJECT_FOLDER.parent

cv = pandas.read_table(folder.parent.joinpath("sup_fig6_npclassifier_cv", "cv.report.tsv"))
cv = cv[['name', 'rank', 'n_positives', 'auprc', 'auroc', 'f1_score']]
cv.columns = ['name', 'rank', 'n_positives', 'auprc', 'auroc', 'f1_score']

cv.to_csv(folder.joinpath('report.tsv'), sep='\t', index=False)
