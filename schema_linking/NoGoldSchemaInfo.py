import random
import os
import sqlite3
import pandas as pd

if __name__ == '__main__':
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema_linking.GoldFilter import GoldFilter
from schema_linking.SchemaInfo import SchemaInfo


class NoGoldSchemaInfo(GoldFilter, SchemaInfo):
    def __init__(self, drop_ratio=0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.drop_ratio = drop_ratio  # 删除比例

    def _get_selected_columns(self, query, table_name, columns_info, **kwargs):
        """Select gold columns, then remove a fraction for the 3_2 condition.

        Keeping this operation in the selection method makes it work for both
        BIRD's description-CSV path and Spider's SQLite-introspection fallback.
        """
        selected_columns = GoldFilter._get_selected_columns(
            self, query, table_name, columns_info, **kwargs
        )
        if self.drop_ratio > 0 and len(selected_columns) > 1:
            keep_num = min(
                len(selected_columns) - 1,
                max(1, int(len(selected_columns) * (1 - self.drop_ratio))),
            )
            selected_columns = random.sample(selected_columns, keep_num)
        return selected_columns

    def _format_columns_description(
            self,
            table_name: str,
            schema_description: pd.DataFrame,
            cursor: sqlite3.Cursor,
            foreign_keys_map,
            **kwargs
    ) -> str:

        columns_schema_str = ""

        question = kwargs.pop("question")
        columns_info = [
            [None, str(row['original_column_name']).strip(), None, None, None, None]
            for _, row in schema_description.iterrows()
        ]

        selected_columns = self._get_selected_columns(
            None,
            table_name,
            columns_info,
            question=question
        )

        if len(selected_columns) == 0:
            return columns_schema_str

        for _, row in schema_description.iterrows():
            column_name = str(row['original_column_name']).strip()
            if column_name not in selected_columns:
                continue

            columns_schema_str += self._format_column_description(
                row, table_name, cursor, foreign_keys_map
            )

        return columns_schema_str
