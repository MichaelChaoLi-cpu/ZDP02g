"""Paired outcome predictions with one fitted model per random fold.

Mediator frames must be generated with a consistent, externally specified first-stage
scheme. This module does not certify nested cross-fitting or causal mediation.
"""
from dataclasses import dataclass
from collections.abc import Callable, Iterable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Perturbation:
    name: str
    feature: str
    delta: float
    mediators: pd.DataFrame


def paired_predictions(
    features: pd.DataFrame,
    target: pd.Series,
    baseline_mediators: pd.DataFrame,
    variants: list[Perturbation],
    splits: Iterable[tuple[np.ndarray, np.ndarray]],
    model_factory: Callable,
    zero_tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Fit on baseline predicted mediators; evaluate all contrasts on that same fit.

    Returns one out-of-fold prediction per input row. Rejects missing, reordered,
    overlapping or incomplete validation rows rather than silently realigning.
    """
    if not features.index.is_unique or not features.columns.is_unique:
        raise ValueError('Feature row and column labels must be unique')
    if not target.index.equals(features.index):
        raise ValueError('Target index/order differs from features')
    if baseline_mediators.empty or not baseline_mediators.columns.is_unique:
        raise ValueError('Expected unique, nonempty mediator columns')
    mediator_columns = list(baseline_mediators.columns)
    if not set(mediator_columns).issubset(features.columns):
        raise ValueError('Mediator columns absent from outcome features')

    def validate_mediators(frame):
        if not frame.index.equals(features.index) or list(frame.columns) != mediator_columns:
            raise ValueError('Mediator index/order or schema mismatch')
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Expected finite binary-mediator probabilities in [0, 1]')

    validate_mediators(baseline_mediators)
    names = [v.name for v in variants]
    if len(set(names)) != len(names) or 'baseline' in names:
        raise ValueError('Variant names must be unique and not baseline')
    for variant in variants:
        validate_mediators(variant.mediators)
        if variant.feature not in features or variant.feature in mediator_columns or not np.isfinite(variant.delta):
            raise ValueError('Invalid perturbation feature/delta')
        if variant.delta == 0 and not np.array_equal(variant.mediators.to_numpy(), baseline_mediators.to_numpy()):
            raise ValueError('Zero perturbation must preserve mediator predictions')
    X = features.copy(deep=True)
    X[mediator_columns] = baseline_mediators
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError('Nonfinite modelling inputs')
    columns = ['baseline', 'fold']
    for name in names:
        columns += [f'{name}_direct', f'{name}_total', f'{name}_direct_change', f'{name}_total_change']
    result = pd.DataFrame(np.nan, index=X.index, columns=columns)
    seen = np.zeros(len(X), dtype=int)
    for fold, (train, test) in enumerate(splits):
        train, test = np.asarray(train), np.asarray(test)
        for indices in (train, test):
            if not np.issubdtype(indices.dtype, np.integer) or len(indices)==0 or len(np.unique(indices))!=len(indices) or indices.min()<0 or indices.max()>=len(X):
                raise ValueError('Invalid fold positions')
        if np.intersect1d(train, test).size or seen[test].any():
            raise ValueError('Overlapping train/test or repeated validation rows')
        model = model_factory()
        model.fit(X.iloc[train], target.iloc[train])
        base = X.iloc[test].copy()
        fold_variants = [Perturbation(v.name, v.feature, v.delta, v.mediators.iloc[test]) for v in variants]
        values = evaluate_paired_fold(model, base, fold_variants, mediator_columns, zero_tolerance)
        for name, value in values.items():
            result.loc[base.index, name] = value
        result.loc[base.index, 'fold'] = fold
        seen[test] += 1
    if not np.all(seen == 1) or not np.isfinite(result.to_numpy()).all():
        raise ValueError('Incomplete/nonfinite out-of-fold predictions')
    return result


def mediator_predictions(model, features: pd.DataFrame, feature: str, delta: float) -> np.ndarray:
    """Use a fitted binary mediator model for baseline or perturbed probabilities."""
    if feature not in features or not np.isfinite(delta):
        raise ValueError('Invalid mediator perturbation')
    classes = np.asarray(model.classes_)
    positive = np.flatnonzero(classes == 1)
    if len(classes) != 2 or len(positive) != 1:
        raise ValueError('Expected binary mediator with positive label 1')
    perturbed = features.copy(deep=True)
    perturbed[feature] = perturbed[feature] + delta
    prediction = np.asarray(model.predict_proba(perturbed))[:, positive[0]]
    if not np.isfinite(prediction).all():
        raise ValueError('Nonfinite mediator predictions')
    return prediction


def evaluate_paired_fold(model, baseline_features, variants, mediator_columns, zero_tolerance=1e-10):
    """Evaluate one fitted fold model; shared by cached and nested first-stage paths."""
    prediction = np.asarray(model.predict(baseline_features))
    result = {'baseline': prediction}
    for variant in variants:
        if not variant.mediators.index.equals(baseline_features.index):
            raise ValueError('Fold mediator index mismatch')
        direct = baseline_features.copy(deep=True)
        direct[variant.feature] = direct[variant.feature] + variant.delta
        total = direct.copy(deep=True)
        total[mediator_columns] = variant.mediators[mediator_columns]
        d, t = np.asarray(model.predict(direct)), np.asarray(model.predict(total))
        if variant.delta == 0 and (not np.allclose(d,prediction,atol=zero_tolerance,rtol=0) or not np.allclose(t,prediction,atol=zero_tolerance,rtol=0)):
            raise ValueError('Zero-perturbation prediction identity failed')
        result.update({f'{variant.name}_direct':d, f'{variant.name}_total':t,
                       f'{variant.name}_direct_change':d-prediction, f'{variant.name}_total_change':t-prediction})
    if not all(np.isfinite(v).all() for v in result.values()):
        raise ValueError('Nonfinite paired predictions')
    return result
