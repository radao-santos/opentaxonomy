from pathlib import Path

import pandas as pd

from .base import DataSource


class CSVSource(DataSource):
    def __init__(self, path: str, sep: str = ","):
        self.path = Path(path)
        self.sep = sep

    def read(self) -> pd.DataFrame:
        return pd.read_csv(self.path, sep=self.sep)

    def write(self, df: pd.DataFrame) -> None:
        df.to_csv(self.path, sep=self.sep, index=False)


class JSONSource(DataSource):
    def __init__(self, path: str):
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        return pd.read_json(self.path)

    def write(self, df: pd.DataFrame) -> None:
        df.to_json(self.path, orient="records", indent=2, force_ascii=False)


class ExcelSource(DataSource):
    def __init__(self, path: str, sheet_name: str | int = 0):
        self.path = Path(path)
        self.sheet_name = sheet_name

    def read(self) -> pd.DataFrame:
        return pd.read_excel(self.path, sheet_name=self.sheet_name)

    def write(self, df: pd.DataFrame) -> None:
        df.to_excel(self.path, sheet_name=self.sheet_name, index=False)


class ParquetSource(DataSource):
    def __init__(self, path: str):
        self.path = Path(path)

    def read(self) -> pd.DataFrame:
        return pd.read_parquet(self.path)

    def write(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.path, index=False)


_EXT_MAP: dict[str, type[DataSource]] = {
    ".csv": CSVSource,
    ".json": JSONSource,
    ".jsonl": JSONSource,
    ".xlsx": ExcelSource,
    ".xls": ExcelSource,
    ".parquet": ParquetSource,
}


def source_from_path(path: str, **kwargs) -> DataSource:
    """Auto-detect DataSource from file extension."""
    ext = Path(path).suffix.lower()
    if ext == ".tsv":
        return CSVSource(path, sep="\t")
    cls = _EXT_MAP.get(ext)
    if cls is None:
        supported = list(_EXT_MAP.keys()) + [".tsv"]
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {supported}")
    return cls(path, **kwargs)
