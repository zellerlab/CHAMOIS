import argparse
import pathlib

from ..predictor import ChemicalOntologyPredictor


def configure_group_training_input(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        'Input',
        'Mandatory input files required by the command.'
    )
    group.add_argument(
        "-f",
        "--features",
        required=True,
        type=pathlib.Path,
        help="The feature table in HDF5 format to use for training the predictor."
    )
    group.add_argument(
        "-c",
        "--classes",
        required=True,
        type=pathlib.Path,
        help="The classes table in HDF5 format to use for training the predictor."
    )
    group.add_argument(
        "-s",
        "--similarity",
        type=pathlib.Path,
        help="Pairwise nucleotide similarities for deduplicating the observations."
    )
    return group


def configure_group_predict_input(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        'Input',
        'Mandatory input files required by the command.'
    )
    group.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        action="append",
        help="The input BGC sequences to process."
    )
    group.add_argument(
        "-H",
        "--hmm",
        type=pathlib.Path,
        help="The path to the HMM file containing protein domains for annotation."
    )
    return group


def configure_group_search_input(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        'Input',
        'Mandatory input files required by the command.'
    )
    group.add_argument(
        "-i",
        "--input",
        required=True,
        type=pathlib.Path,
        help="The chemical classes predicted by CONCH for BGCs."
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        type=pathlib.Path,
        help="The path to an alternative model used for predicting classes."
    )
    return group


def configure_group_search_parameters(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        "Search",
        "Parameters for controling the search procedure and result filtering."
    )
    group.add_argument(
        "-d",
        "--distance",
        default="hamming",
        help="The metric to use for comparing classes fingerprints.",
        choices={"hamming", "jaccard"},
    )
    return group


def configure_group_gene_finding(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        "Gene Finding",
        "Parameters for controlling gene extraction from clusters."
    )
    group.add_argument(
        "--cds",
        action="store_true",
        help="Use CDS features in the GenBank input as genes instead of running Pyrodigal.",
    )
    return group


def configure_group_preprocessing(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        'Preprocessing',
        'Parameters controling data preprocessing, including features and labels filtering.'
    )
    group.add_argument(
        "--min-class-occurrences",
        type=int,
        default=10,
        help="The minimum of occurences for a class to be retained."
    )
    group.add_argument(
        "--min-feature-occurrences",
        type=int,
        default=0,
        help="The minimum of occurences for a feature to be retained."
    )
    group.add_argument(
        "--min-cluster-length",
        type=int,
        default=1000,
        help="The nucleotide length threshold for retaining a cluster."
    )
    group.add_argument(
        "--min-genes",
        type=int,
        default=2,
        help="The gene count threshold for retaining a cluster."
    )
    return group


def configure_group_hyperparameters(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        'Training',
        'Hyperparameters to use for training the model.'
    )
    group.add_argument(
        "--model",
        choices=ChemicalOntologyPredictor._MODELS,
        default="logistic",
        help="The kind of model to train."
    )
    group.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="The strength of the parameters regularization.",
    )
    group.add_argument(
        "--variance",
        type=float,
        help="The variance threshold for filtering features.",
        default=None,
    )
    return group


def configure_group_cross_validation(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        'Cross-validation',
        'Parameters controlling the cross-validation.'
    )
    group.add_argument(
        "-k",
        "--kfolds",
        type=int,
        default=10,
        help="The number of cross-validation folds to run.",
    )
    group.add_argument(
        "--sampling",
        choices={"random", "group", "kennard-stone"},
        default="group",
        help="The algorithm to use for partitioning folds.",
    )
    return group


def configure_group_search_output(parser: argparse.ArgumentParser) -> "argparse.ArgumentGroup":
    group = parser.add_argument_group(
        "Output",
        "Parameters for controlling command output."
    )
    group.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="The path where to write the catalog search results in TSV format."
    )
    group.add_argument(
        "--rank",
        default=10,
        type=int,
        help="The maximum search rank to record in the table output.",
    )
    group.add_argument(
        "--render",
        action="store_true",
        help="Display best match for each query.",
    )
    return group