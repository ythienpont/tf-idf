import math

from tfidf.schemes.idf_schemes import (
    IdfUnaryScheme, IdfStandardScheme, IdfSmoothScheme,
    IdfMaxScheme, IdfProbScheme
)

def test_idf_unary_scheme():
    assert IdfUnaryScheme.calculate() == 1.0

def test_idf_standard_scheme():
    assert IdfStandardScheme.calculate(10, 2) == math.log(10 / 2)
    assert IdfStandardScheme.calculate(10, 10) == 0.0
    assert IdfStandardScheme.calculate(10, 0) == 0.0

def test_idf_smooth_scheme():
    assert IdfSmoothScheme.calculate(10, 2) == math.log((10 + 1) / (2 + 1))
    assert IdfSmoothScheme.calculate(10, 10) == math.log((10 + 1) / (10 + 1))
    assert IdfSmoothScheme.calculate(10, 0) == math.log(11/1)

def test_idf_max_scheme():
    assert IdfMaxScheme.calculate(10, 2, 10) == math.log(10 / (2 + 1))
    assert IdfMaxScheme.calculate(10, 10, 10) == math.log(10 / (10 + 1))
    assert IdfMaxScheme.calculate(10, 0, 10) == 0.0

def test_idf_prob_scheme():
    assert IdfProbScheme.calculate(10, 2) == math.log((10 - 2) / 2)
    assert IdfProbScheme.calculate(10, 10) == 0.0
    assert IdfProbScheme.calculate(10, 0) == 0.0
