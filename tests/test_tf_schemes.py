import pytest
import math

from tfidf.schemes.tf_schemes import (
    TfBinaryScheme, TfRawScheme, TfStandardScheme,
    TfLogNormScheme, TfDoubleNormScheme
)

def test_tf_binary_scheme_positive():
    assert TfBinaryScheme.calculate(3) == 1.0
    assert TfBinaryScheme.calculate(0) == 0.0

def test_tf_binary_scheme_zero():
    assert TfBinaryScheme.calculate(0) == 0.0

def test_tf_raw_scheme():
    assert TfRawScheme.calculate(5) == 5.0
    assert TfRawScheme.calculate(0) == 0.0
    assert TfRawScheme.calculate(10) == 10.0

def test_tf_standard_scheme():
    assert TfStandardScheme.calculate(3, total=10) == 0.3
    assert TfStandardScheme.calculate(5, total=20) == 0.25
    with pytest.raises(ZeroDivisionError):
        TfStandardScheme.calculate(5, total=0)

def test_tf_log_norm_scheme():
    assert TfLogNormScheme.calculate(0) == pytest.approx(0.0)
    assert TfLogNormScheme.calculate(1) == pytest.approx(math.log(2))
    assert TfLogNormScheme.calculate(9) == pytest.approx(math.log(10))

def test_tf_double_norm_scheme():
    assert TfDoubleNormScheme.calculate(3, max_count=10) == pytest.approx(0.65)
    assert TfDoubleNormScheme.calculate(5, max_count=5) == pytest.approx(1.0)
    assert TfDoubleNormScheme.calculate(0, max_count=5) == 0.5
    assert TfDoubleNormScheme.calculate(4, max_count=0) == 0.0
