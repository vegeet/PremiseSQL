from .SchemaLinker import SchemaLinker

import importlib

# 动态import schema_linker
def load_schema_linker(schema_linking_strategy: str, **kwargs) -> SchemaLinker:
    schema_linking_module = importlib.import_module('schema_linking.' + schema_linking_strategy)
    schema_linker = getattr(schema_linking_module, schema_linking_strategy)(**kwargs)
    return schema_linker
