"""Auto-discover and import all provider modules.

Dropping a new ``xxx_provider.py`` into this directory is all that is
needed to register a new capability provider — ``pkgutil.iter_modules``
discovers it and the ``@register_provider`` decorator fires at import
time.
"""

import importlib
import pkgutil

for _, module_name, _ in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)
