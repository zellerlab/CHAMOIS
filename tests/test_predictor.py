import unittest
import warnings

import numpy
import pandas

from chamois.model import ClusterSequence, Protein
from chamois.predictor import ChemicalOntologyPredictor
from chamois.ontology import Ontology

from .utils import resource_files


class TestChemicalOntologyPredictor(unittest.TestCase):
    
    def test_predict(self):
        # check the predictions for BGC0000703 are consistent 
        # (this is using the sparse domain representation
        # extracted from data/datasets/mibig3.1/features.hdf5)
        predictor = ChemicalOntologyPredictor.trained()
        X = numpy.zeros((1, len(predictor.features_)), dtype=bool)
        features = [   
            1,   23,   26,   46,   70,  141,  150,  170,  187,  224,  227,
            284,  297,  298,  367,  373,  417,  450,  474,  490,  553,  558,
            586,  631,  662,  669,  689,  690,  709,  734,  779,  845,  866,
            904,  928,  948,  995, 1095, 1201, 1250, 1254, 1285, 1286, 1308,
            1310, 1322, 1337, 1339, 1437, 1572, 1585
        ]
        X[0, features] = 1
        labels = predictor.predict(X)[0]
        self.assertEqual(
            list(numpy.where(labels)[0]),
            [  
                0,   3,  42,  79,  89,  91,  94, 102, 129, 185, 206, 219, 223,
                230, 253, 268, 273, 279, 290, 310, 311, 316, 327, 328, 340, 342,
                379, 394, 432, 437, 470, 471, 477
            ]
        )

    def test_fit(self):
        ontology = Ontology(numpy.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ]))
        predictor = ChemicalOntologyPredictor(ontology=ontology)
        X = numpy.random.binomial(1, 0.5, (20, 20))
        Y = numpy.random.binomial(1, 0.5, (20, 5))
        predictor.fit(X, Y)