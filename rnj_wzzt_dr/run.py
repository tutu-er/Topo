"""Run from any directory, including a renamed copy under Python -I."""
from pathlib import Path
import importlib
import importlib.util
import sys


def main():
    if __package__:
        from .cli import main as run
    else:
        root = Path(__file__).resolve().parent
        key = "_rnj_wzzt_dr_local"
        spec = importlib.util.spec_from_file_location(key, root/"__init__.py",
                                                     submodule_search_locations=[str(root)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[key] = package
        spec.loader.exec_module(package)
        run = importlib.import_module(key+".cli").main
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
