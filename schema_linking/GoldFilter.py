import json
from typing import List

from schema_linking import SchemaLinker
from schema_linking.VanillaLinker import VanillaLinker


class GoldFilter(SchemaLinker):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        gold_schema_path = kwargs.pop(
            "gold_schema_path",
            "data/bird/dev_gold_schema.json"
            # "data/spider/dev_gold_schema.json"
        )
        with open(gold_schema_path, "r", encoding="utf-8") as f:
            self.gold_schema = json.load(f)

    # def _get_selected_columns(self, query: str, table_name: str, columns_info: List, **kwargs) -> List[str]:
    #     try:
    #         question = kwargs.pop("question")
    #         return self.gold_schema[question].get(table_name, [])
    #     except:
    #         print("Bad News: We are into VanillaLinker!")
    #         return VanillaLinker._get_selected_columns(self, query, table_name, columns_info, **kwargs)

    def _get_selected_columns(self, query, table_name, columns_info, **kwargs):
        question = kwargs.get("question")

        if question not in self.gold_schema:
            print(f"[Skip] No gold schema for: {question}")
            return []

        return self.gold_schema[question].get(table_name, [])


