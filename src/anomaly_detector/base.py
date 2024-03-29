# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from abc import abstractmethod

import mlflow


class BaseAnomalyDetector(mlflow.pyfunc.PythonModel):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, context, model_input, params=None):
        pass
