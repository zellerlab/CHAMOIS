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
            1, 14, 18, 32, 46, 92, 112, 122, 149, 150, 185, 194, 195, 230,
            234, 259, 281, 297, 333, 336, 349, 370, 401, 402, 408, 421, 439,
            469, 480, 499, 506, 516, 544, 591, 644, 674, 677, 702, 718, 720, 
            730, 741, 780, 827, 833, 861, 880
        ]
        X[0, features] = 1
        labels = predictor.predict(X)[0]
        self.assertEqual(
            list(numpy.where(labels)[0]),
            [
                0,   3,  46,  87,  98,  99, 110, 137, 204, 229, 246, 250, 255,
               257, 279, 297, 305, 312, 322, 346, 347, 352, 363, 364, 378, 380,
               425, 478, 483, 519, 520, 528,
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
