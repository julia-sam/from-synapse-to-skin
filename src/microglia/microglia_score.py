from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


MICROGLIA_CORE_MARKERS = [
    "P2RY12",
    "CX3CR1",
    "TMEM119",
    "AIF1",
    "ITGAM",
    "SPI1",
    "TREM2",
    "TYROBP",
    "LST1",
    "C1QA",
    "C1QB",
    "C1QC",
]

# A small, generic inflammatory/activation-biased set (edit/replace for your biology).
MICROGLIA_ACTIVATION_UP = [
    "IL1B",
    "TNF",
    "NFKBIA",
    "CXCL10",
    "CCL2",
    "IFITM3",
    "HLA-DRA",
    "HLA-DRB1",
    "S100A8",
    "S100A9",
]

MICROGLIA_INFLAMMATION_UP = [
    "IL1B",
    "TNF",
    "IL6",
    "NFKBIA",
    "NFKBIZ",
    "PTGS2",
    "CCL2",
    "CCL3",
    "CCL4",
    "CXCL10",
    "CXCL8",
    "IRF1",
    "STAT1",
    "SOCS3",
]


@dataclass(frozen=True)
class SignatureScoreResult:
    score: pd.Series
    genes_used: list[str]
    genes_missing: list[str]


def _normalize_gene_symbols(gene_symbols: Iterable[str]) -> list[str]:
    out = []
    for g in gene_symbols:
        if g is None:
            continue
        g = str(g).strip()
        if not g:
            continue
        out.append(g.upper())
    return out


def _standardize_index_to_symbols(expr: pd.DataFrame) -> pd.DataFrame:
    out = expr.copy()
    out.index = out.index.astype(str).str.upper()
    return out


def score_signature_mean_z(
    expression: pd.DataFrame,
    genes: Iterable[str],
    *,
    min_genes: int = 5,
) -> SignatureScoreResult:
    """
    Compute a simple per-sample signature score:
      1) z-score each gene across samples
      2) average z-scores across the provided gene list

    `expression` must be genes x samples (index = gene symbol-like identifiers).
    """

    if expression.empty:
        raise ValueError("expression is empty")

    expr = _standardize_index_to_symbols(expression)
    gene_list = _normalize_gene_symbols(genes)

    present = [g for g in gene_list if g in expr.index]
    missing = [g for g in gene_list if g not in expr.index]

    if len(present) < min_genes:
        looks_like_ensembl = expr.index.astype(str).str.startswith("ENSG").mean() > 0.5
        looks_like_symbols = (
            len(gene_list) > 0
            and sum((not g.startswith("ENSG")) and all(ch.isalnum() or ch in {"-", "."} for ch in g) for g in gene_list)
            / max(len(gene_list), 1)
            > 0.5
        )
        extra_hint = ""
        if looks_like_ensembl and looks_like_symbols:
            extra_hint = (
                " Your expression index looks like Ensembl gene IDs (ENSG...), but your signature is gene symbols. "
                "Provide an Ensembl→symbol mapping, or use a data-driven score like `score_top_correlated(...)`."
            )
        raise ValueError(
            f"Only {len(present)} / {len(gene_list)} genes present in expression (min_genes={min_genes}). "
            f"Missing examples: {missing[:10]}."
            f"{extra_hint}"
        )

    sub = expr.loc[present]
    means = sub.mean(axis=1)
    stds = sub.std(axis=1).replace(0, np.nan)
    z = sub.sub(means, axis=0).div(stds, axis=0)

    score = z.mean(axis=0)
    score.name = "signature_score"

    return SignatureScoreResult(score=score, genes_used=present, genes_missing=missing)


def score_up_minus_down(
    expression: pd.DataFrame,
    up_genes: Iterable[str],
    down_genes: Iterable[str],
    *,
    min_genes: int = 5,
) -> SignatureScoreResult:
    """
    Score = mean_z(up_genes) - mean_z(down_genes).
    """

    up = score_signature_mean_z(expression, up_genes, min_genes=min_genes)
    down = score_signature_mean_z(expression, down_genes, min_genes=min_genes)

    score = up.score - down.score
    score.name = "signature_up_minus_down"

    return SignatureScoreResult(
        score=score,
        genes_used=sorted(set(up.genes_used + down.genes_used)),
        genes_missing=sorted(set(up.genes_missing + down.genes_missing)),
    )


def microglia_activation_score(
    expression: pd.DataFrame,
    *,
    activation_genes: Iterable[str] = MICROGLIA_ACTIVATION_UP,
    min_genes: int = 5,
) -> SignatureScoreResult:
    """
    Convenience wrapper for a "microglia activation" score using `MICROGLIA_ACTIVATION_UP`.
    """

    return score_signature_mean_z(expression, activation_genes, min_genes=min_genes)


def microglia_inflammation_score(
    expression: pd.DataFrame,
    *,
    inflammation_genes: Iterable[str] = MICROGLIA_INFLAMMATION_UP,
    min_genes: int = 5,
) -> SignatureScoreResult:
    """
    Convenience wrapper for an inflammation-biased score.
    """

    return score_signature_mean_z(expression, inflammation_genes, min_genes=min_genes)


def residualize_against_age(
    score: pd.Series,
    age: pd.Series,
    *,
    standardize: bool = True,
) -> pd.Series:
    """
    Remove linear age effects from a per-sample score: score ~ age.
    Returns residuals (optionally z-scored).
    """

    y = pd.to_numeric(score, errors="coerce")
    x = pd.to_numeric(age, errors="coerce")
    df = pd.DataFrame({"score": y, "age": x}).dropna()
    if df.empty or df["age"].nunique() < 2:
        raise ValueError("Not enough age variation to residualize.")

    X = np.column_stack([np.ones(len(df)), df["age"].to_numpy(dtype=float)])
    beta, *_ = np.linalg.lstsq(X, df["score"].to_numpy(dtype=float), rcond=None)
    fitted = X @ beta
    resid = df["score"].to_numpy(dtype=float) - fitted

    out = pd.Series(index=score.index, dtype=float, name=f"{score.name or 'score'}_age_resid")
    out.loc[df.index] = resid
    if standardize:
        mu = out.loc[df.index].mean()
        sd = out.loc[df.index].std()
        if sd and sd > 0:
            out.loc[df.index] = (out.loc[df.index] - mu) / sd
    return out


def score_top_correlated(
    expression: pd.DataFrame,
    phenotype: pd.Series,
    *,
    n_top: int = 100,
    method: str = "spearman",
    direction: str = "positive",
    min_samples: int = 10,
) -> SignatureScoreResult:
    """
    Data-driven signature score:
      1) compute per-gene correlation with a phenotype across samples
      2) take the top-N genes (positive or negative direction)
      3) return mean z-score signature across those genes

    This is useful when your expression index uses Ensembl IDs (e.g. ENSG...)
    and you don't yet have a gene-symbol mapping for curated signatures.

    Parameters
    ----------
    expression:
        DataFrame with genes as index and samples as columns.
    phenotype:
        Series indexed by sample ids. Will be aligned to expression columns.
    method:
        "pearson" or "spearman"
    direction:
        "positive" or "negative"
    """

    if expression.empty:
        raise ValueError("expression is empty")

    if method not in {"pearson", "spearman"}:
        raise ValueError(f"Unsupported method: {method}")
    if direction not in {"positive", "negative"}:
        raise ValueError(f"Unsupported direction: {direction}")

    y = phenotype.reindex(expression.columns)
    y = pd.to_numeric(y, errors="coerce")
    keep = y.notna()
    if keep.sum() < min_samples:
        raise ValueError(f"Not enough phenotype samples after alignment (n={keep.sum()}, min_samples={min_samples})")

    expr = expression.loc[:, keep]
    y = y.loc[keep]

    X = expr.astype(float)
    if method == "spearman":
        X = X.rank(axis=1, method="average", na_option="keep")
        y = y.rank(method="average")

    # z-score across samples
    X_mean = X.mean(axis=1)
    X_std = X.std(axis=1).replace(0, np.nan)
    X_z = X.sub(X_mean, axis=0).div(X_std, axis=0)

    y_z = (y - y.mean()) / (y.std() if y.std() != 0 else np.nan)
    corrs = (X_z.mul(y_z, axis=1)).mean(axis=1)

    corrs = corrs.dropna()
    if corrs.empty:
        raise ValueError("All correlations are NaN after preprocessing")

    if direction == "positive":
        genes_used = corrs.sort_values(ascending=False).head(n_top).index.tolist()
    else:
        genes_used = corrs.sort_values(ascending=True).head(n_top).index.tolist()

    if not genes_used:
        raise ValueError("No genes selected for signature")

    score = X_z.loc[genes_used].mean(axis=0)
    score.name = f"top_{n_top}_{direction}_{method}_corr_signature"

    return SignatureScoreResult(score=score, genes_used=genes_used, genes_missing=[])
