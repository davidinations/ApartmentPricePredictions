"""
Custom transformer classes used inside the preprocessing pipeline that was
pickled into final_model.pkl (see Modeling.ipynb).

These class *bodies* must match Modeling.ipynb exactly -- they're not
re-run on load, but pickle needs to find a class with a compatible
__setstate__ under this exact name.

They also need to be resolvable as `__main__.<ClassName>` when the model
is unpickled, because that's the module they lived in when the pipeline
was originally pickled (a notebook's kernel runs as __main__). The block
at the bottom handles that registration directly, so it works no matter
how this file itself is imported/run (plain script, Streamlit, uvicorn, etc).
"""
import sys
from typing import Literal

import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin


class HandleOutlier(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Size(sqf)'] = winsorize(X['Size(sqf)'], limits=(0.01, 0.01))
        return X

    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)


class AgeTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['AgeProperty'] = 2016 - X['YearBuilt']
        return X

    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)


class AgeBinner(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Binned_AgeProperty'] = pd.cut(
            X['AgeProperty'], bins=[0, 20, 35, 50], labels=[3, 2, 1])
        return X

    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)


# --- Pickle compatibility: register these under whatever __main__ is ----
# right now, in THIS process -- not the module this file happens to be
# imported as. This is what makes unpickling work regardless of whether
# you're running `python app.py`, `streamlit run app.py`, or `uvicorn app:app`.
_main_module = sys.modules.get('__main__')
if _main_module is not None:
    for _cls in (HandleOutlier, AgeTransformer, AgeBinner, ColumnDropper):
        setattr(_main_module, _cls.__name__, _cls)
