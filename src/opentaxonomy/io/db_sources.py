import pandas as pd
from sqlalchemy import create_engine

from .base import DataSource


class SQLSource(DataSource):
    """
    Reads from and writes to a SQL table via SQLAlchemy.

    connection_string examples:
      postgresql://user:pass@host:5432/dbname
      mysql+pymysql://user:pass@host:3306/dbname
      sqlite:///path/to/db.sqlite
      mssql+pyodbc://user:pass@dsn
    """

    def __init__(self, connection_string: str, table: str, schema: str | None = None):
        self.connection_string = connection_string
        self.table = table
        self.schema = schema
        self._engine = None

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(self.connection_string)
        return self._engine

    def read(self) -> pd.DataFrame:
        return pd.read_sql_table(self.table, self.engine, schema=self.schema)

    def write(self, df: pd.DataFrame) -> None:
        df.to_sql(
            self.table,
            self.engine,
            schema=self.schema,
            if_exists="replace",
            index=False,
        )
