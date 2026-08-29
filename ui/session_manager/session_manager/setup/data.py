import json
from pathlib import Path
from typing import List
from functools import wraps

import psycopg2 as pg

from session_manager.data import SQLParams


def reconnect_on_failure(method):
    """Decorator to reconnect and retry once if a connection error occurs"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except (pg.OperationalError, pg.InterfaceError):
            self._init_sql_client()
            return method(self, *args, **kwargs)
    return wrapper


class SetupDB:
    """
    Generic database operations for experiment setup.
    Works with any table - sessions, participants, dialogues, logs, etc.
    """

    def __init__(self, sql_params: SQLParams):
        self.sql_params = sql_params
        self._init_sql_client()

    def _init_sql_client(self):
        """Initialize SQL client connection"""
        self.sql = pg.connect(**self.sql_params.model_dump(mode='json'))
        self.sql.autocommit = True

    @reconnect_on_failure
    def empty_table(self, table: str):
        """Delete all rows from a table."""
        cursor = self.sql.cursor()
        cursor.execute(f"DELETE FROM {table}")
        cursor.close()

    @reconnect_on_failure
    def get_table(self, table: str) -> List[dict]:
        """Get all rows from a table as list of dicts."""
        cursor = self.sql.cursor()
        cursor.execute(f"SELECT * FROM {table}")
        columns = [desc[0] for desc in cursor.description]
        rows = []
        for row in cursor.fetchall():
            rows.append(dict(zip(columns, row)))
        cursor.close()
        return rows

    @reconnect_on_failure
    def fill_table(self, table: str, rows: List[dict]):
        """Insert multiple rows into a table."""        
        cursor = self.sql.cursor()
        columns = rows[0].keys()
        values_str = ', '.join(['%s'] * len(columns))
        insert_query = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({values_str});"
        for row in rows:
            values = []
            for col in columns:
                val = row[col]
                # Convert dicts to JSON strings for jsonb columns
                if col.endswith('_data'):
                    values.append(json.dumps(val))
                else:
                    values.append(val)
            cursor.execute(insert_query, tuple(values))
        cursor.close()
        
    @reconnect_on_failure
    def list_tables(self) -> List[str]:
        """List all tables in the current database schema."""
        cursor = self.sql.cursor()
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables

    def replace_table(self, table: str, rows: List[dict]):
        """Empty table and fill with new rows."""
        self.empty_table(table)
        self.fill_table(table, rows)

    def save_table(self, data: List[dict], table_name: str, path: Path):
        """Save table data to a JSON file."""
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{table_name}.json"
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def download_and_empty_all_tables(self, tables: List[str], backup_dir: Path):
        """Backup all specified tables to JSON files, then empty them."""
        for table in tables:
            data = self.get_table(table)
            self.save_table(data, table, backup_dir)
            self.empty_table(table)
