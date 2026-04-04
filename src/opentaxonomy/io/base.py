from abc import ABC, abstractmethod

import pandas as pd


class DataSource(ABC):
    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Read the full dataset as a DataFrame."""
        ...

    @abstractmethod
    def write(self, df: pd.DataFrame) -> None:
        """Write enriched DataFrame back to the source."""
        ...
