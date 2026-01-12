from __future__ import annotations

from dataclasses import dataclass
import gzip
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd
import json
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class GeoDataset:
    """
    Standard in-memory representation for a GEO dataset.

    expression:
        DataFrame with genes as index and samples as columns.
    samples:
        DataFrame indexed by sample accession (e.g. GSM...), with parsed metadata columns.
    series:
        Top-level GSE metadata, as key -> list[str] from SOFT where available.
    """

    gse_id: str
    expression: pd.DataFrame
    samples: pd.DataFrame
    series: dict[str, list[str]]


def _open_text(path: Path) -> IO[str]:
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", errors="replace")
    return path.open("r", errors="replace")


def parse_family_soft(soft_path: Path) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """
    Parse a GEO "family" SOFT file into (series_metadata, samples_df).

    This is intentionally lightweight (no GEOparse dependency) and only extracts:
    - !Series_* fields (stored as dict[str, list[str]])
    - !Sample_* fields (flattened into columns)
    - !Sample_characteristics_ch1 = key: value lines (expanded into columns)
    """

    series: dict[str, list[str]] = {}
    samples: dict[str, dict[str, Any]] = {}

    current_sample: str | None = None

    def add_series(key: str, value: str) -> None:
        series.setdefault(key, []).append(value)

    def norm_key(key: str) -> str:
        return (
            key.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )

    with _open_text(soft_path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            if line.startswith("^SAMPLE = "):
                current_sample = line.split("=", 1)[1].strip()
                samples.setdefault(current_sample, {"gsm": current_sample})
                continue

            if line.startswith("^SERIES = "):
                current_sample = None
                continue

            if line.startswith("!Series_"):
                k, v = line.split("=", 1)
                add_series(norm_key(k.lstrip("!")), v.strip())
                continue

            if current_sample is None:
                continue

            if line.startswith("!Sample_"):
                k, v = line.split("=", 1)
                key = norm_key(k.lstrip("!"))
                value = v.strip()

                if key == "sample_characteristics_ch1":
                    # e.g. "age: 78"
                    if ":" in value:
                        c_key, c_val = value.split(":", 1)
                        c_key = norm_key(c_key)
                        c_val = c_val.strip()
                        prior = samples[current_sample].get(c_key)
                        if prior is None:
                            samples[current_sample][c_key] = c_val
                        elif isinstance(prior, list):
                            prior.append(c_val)
                        else:
                            samples[current_sample][c_key] = [prior, c_val]
                else:
                    # store first occurrence; if repeated, keep list
                    prior = samples[current_sample].get(key)
                    if prior is None:
                        samples[current_sample][key] = value
                    elif isinstance(prior, list):
                        prior.append(value)
                    else:
                        samples[current_sample][key] = [prior, value]

    samples_df = pd.DataFrame.from_dict(samples, orient="index")
    if "gsm" in samples_df.columns:
        samples_df = samples_df.set_index("gsm", drop=True)
    samples_df.index.name = "gsm"

    return series, samples_df


def load_counts_table(counts_path: Path) -> pd.DataFrame:
    """
    Load a supplementary gene-count table (typically tab-separated).

    Returns a DataFrame with gene identifiers as index and sample columns (e.g. GSM...).
    """

    if not counts_path.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_path}")

    df = pd.read_csv(counts_path, sep="\t", dtype=str)
    if df.shape[1] < 2:
        raise ValueError(f"Counts file looks empty or malformed: {counts_path}")

    preferred_gene_cols = [
        "gene",
        "genes",
        "gene_symbol",
        "symbol",
        "genesymbol",
        "geneid",
        "ensembl",
        "id",
        "id_ref",
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    gene_col = None
    for c in preferred_gene_cols:
        if c in lower_cols:
            gene_col = lower_cols[c]
            break
    if gene_col is None:
        gene_col = df.columns[0]

    df = df.rename(columns={gene_col: "gene"})
    df["gene"] = df["gene"].astype(str)
    df = df.set_index("gene", drop=True)
    df.index = df.index.astype(str)

    # coerce sample columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # collapse duplicate gene identifiers (common when mixing Ensembl versions)
    if df.index.duplicated().any():
        df = df.groupby(level=0).sum(numeric_only=True)

    return df


def log_cpm(counts: pd.DataFrame, pseudocount: float = 1.0) -> pd.DataFrame:
    """
    Convert raw counts to log(CPM + pseudocount). Assumes genes x samples.
    """

    lib_sizes = counts.sum(axis=0)
    if (lib_sizes <= 0).any():
        bad = lib_sizes[lib_sizes <= 0]
        raise ValueError(f"Non-positive library sizes found (cannot compute CPM): {bad.to_dict()}")
    cpm = counts.div(lib_sizes, axis=1) * 1_000_000
    return np.log(cpm + float(pseudocount))


def strip_ensembl_version(gene_id: str) -> str:
    """
    Convert e.g. 'ENSG000001234.5' -> 'ENSG000001234'.
    """

    gene_id = str(gene_id).strip()
    if "." in gene_id:
        head, tail = gene_id.split(".", 1)
        if head.startswith("ENSG") and tail.isdigit():
            return head
    return gene_id


def fetch_ensembl_to_symbol_map(
    ensembl_ids: list[str],
    *,
    species: str = "human",
    batch_size: int = 1000,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """
    Fetch Ensembl Gene ID -> HGNC symbol mapping from mygene.info.

    Requires network access at runtime.
    """

    if not ensembl_ids:
        return {}

    cleaned = [strip_ensembl_version(e) for e in ensembl_ids]
    unique = list(dict.fromkeys(cleaned))

    out: dict[str, str] = {}
    url = "https://mygene.info/v3/gene"

    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        payload = {
            "ids": batch,
            "species": species,
            "fields": ["symbol", "ensembl.gene"],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            rows = json.loads(resp.read().decode("utf-8"))

        for row in rows:
            if not isinstance(row, dict):
                continue
            q = row.get("query")
            symbol = row.get("symbol")
            if q and symbol and isinstance(symbol, str):
                out[str(q)] = symbol

    return out


def convert_ensembl_index_to_symbols(
    expression: pd.DataFrame,
    ensembl_to_symbol: dict[str, str],
    *,
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Convert an expression matrix indexed by Ensembl IDs (ENSG...) to gene symbols.

    - Strips Ensembl version suffixes before mapping.
    - Aggregates duplicate symbols by summing.
    """

    if expression.empty:
        return expression

    expr = expression.copy()
    ensg = expr.index.astype(str).map(strip_ensembl_version)
    symbols = ensg.map(lambda x: ensembl_to_symbol.get(x))
    expr.index = symbols
    expr.index.name = "gene_symbol"

    if drop_unmapped:
        expr = expr.loc[expr.index.notna()]

    # If multiple Ensembl IDs map to same symbol, aggregate.
    if expr.index.duplicated().any():
        expr = expr.groupby(level=0).sum(numeric_only=True)

    # Standardize casing
    expr.index = expr.index.astype(str).str.upper()
    return expr


def load_gse99074(
    root: Path | None = None,
    counts_filename: str = "GSE99074_HumanMicrogliaBrainCounts.txt.gz",
    soft_filename: str = "GSE99074_family.soft.gz",
) -> GeoDataset:
    """
    Load GSE99074 (human microglia / brain) from local `data/` files.

    Expected layout:
      data/microglia_GSE99074/raw/<soft_filename>
      data/microglia_GSE99074/raw/<counts_filename>
    """

    if root is None:
        root = DATA_ROOT / "microglia_GSE99074"

    raw_dir = root / "raw"
    soft_path = raw_dir / soft_filename
    counts_path = raw_dir / counts_filename

    if not soft_path.exists():
        raise FileNotFoundError(
            f"Missing SOFT file: {soft_path}\n"
            f"Tip: save the family SOFT as {soft_filename} under {raw_dir}"
        )

    if not counts_path.exists():
        raise FileNotFoundError(
            f"Missing counts file: {counts_path}\n"
            "GSE99074 provides counts as a supplementary file named\n"
            f"  {counts_filename}\n"
            "Download it from GEO and place it under the `raw/` folder."
        )

    series_meta, samples_df = parse_family_soft(soft_path)
    counts_df = load_counts_table(counts_path)

    # Align expression columns to sample metadata.
    # In GSE99074, counts columns are typically sample titles (e.g. "spm09", "S291"),
    # while SOFT metadata is indexed by GSM accession. Map title -> GSM so downstream
    # code can consistently use GSM ids.
    if "sample_title" in samples_df.columns:
        title_to_gsm = (
            samples_df["sample_title"]
            .dropna()
            .astype(str)
            .to_frame("sample_title")
            .reset_index()
            .drop_duplicates(subset=["sample_title"], keep="first")
            .set_index("sample_title")["gsm"]
            .to_dict()
        )
        counts_df = counts_df.rename(columns={c: title_to_gsm.get(str(c), c) for c in counts_df.columns})

    # Keep only GSM columns when present, and order to match metadata.
    gsm_cols = [c for c in counts_df.columns if str(c).startswith("GSM")]
    if gsm_cols:
        # If some GSMs aren't in metadata, keep them but we can't annotate them.
        counts_df = counts_df[gsm_cols]

    common = [c for c in counts_df.columns if c in samples_df.index]
    if common:
        counts_df = counts_df[common]
        samples_df = samples_df.loc[common]

    return GeoDataset(gse_id="GSE99074", expression=counts_df, samples=samples_df, series=series_meta)
