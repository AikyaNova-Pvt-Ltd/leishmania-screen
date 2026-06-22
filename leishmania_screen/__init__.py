"""
leishmania-screen
=================
Neural network–based virtual screening for antileishmanial activity
against *Leishmania donovani*.

Quick start
-----------
>>> from leishmania_screen import predict
>>> result = predict("CCO")
>>> print(result.label, result.probability)

>>> results = predict(["CCO", "c1ccccc1"])
"""

from ._predict import predict, PredictionResult, THRESHOLD

__version__ = "1.0.3"
__author__  = "Madhavi et al."
__all__     = ["predict", "PredictionResult", "THRESHOLD"]
