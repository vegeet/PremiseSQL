from typing import List

if __name__ == '__main__':
    import os, sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schema_linking import SchemaLinker


class VanillaLinker(SchemaLinker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _get_selected_columns(self, query: str, table_name: str, columns_info: List, **kwargs) -> List[str]:
        column_names = [column_info[1] for column_info in columns_info]
        return column_names



