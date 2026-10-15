from .Base import Base

import importlib

def load_agent(method: str, args) -> Base:
    method_module = importlib.import_module('model.' + method)
    agent = getattr(method_module, method)(args)
    return agent
    
