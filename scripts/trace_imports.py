import builtins

_real_import = builtins.__import__
seen = set()

def trace_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in seen and (name.startswith("insightface") or name.startswith("onnxruntime") or name.startswith("torch") or name.startswith("faiss")):
        print(f"Importing {name}")
        seen.add(name)
    return _real_import(name, globals, locals, fromlist, level)

builtins.__import__ = trace_import

import insightface