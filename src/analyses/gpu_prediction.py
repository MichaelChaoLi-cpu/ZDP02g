"""Keep fitting unchanged while moving numeric prediction inputs onto the GPU.

Optional CuPy dependency is installed with ``uv sync --extra gpu`` on Linux.
Column validation happens before conversion, because arrays carry no feature names.
"""
import numpy as np


class GPUPredictionModel:
    def __init__(self, model):
        import cupy as cp
        self.model = model
        self.cp = cp
        device = str(model.get_params().get('device', 'cpu'))
        if not device.startswith('cuda:'):
            raise ValueError('An explicit CUDA device is required')
        self.device_id = int(device.split(':')[1])

    def __getattr__(self, name):
        return getattr(self.model, name)

    def fit(self, X, y, **kwargs):
        self.columns = list(X.columns)
        self.model.fit(X, y, **kwargs)
        return self

    def _predict(self, method, X):
        if list(X.columns) != self.columns:
            raise ValueError('Prediction feature names/order differ from fitted data')
        if not all(dtype.kind in 'biuf' for dtype in X.dtypes):
            raise ValueError('GPU prediction requires numeric features')
        # XGBoost numeric inputs use float32 internally; keep the fitted estimator's
        # own predict methods for objective transformation and class handling.
        with self.cp.cuda.Device(self.device_id):
            values = self.cp.asarray(X.to_numpy(dtype=np.float32))
            return getattr(self.model, method)(values)

    def predict(self, X):
        return self._predict('predict', X)

    def predict_proba(self, X):
        return self._predict('predict_proba', X)
