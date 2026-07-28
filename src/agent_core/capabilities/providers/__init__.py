"""自动发现并导入所有 Provider 模块。

注册新能力 Provider 时只需在本目录添加 ``xxx_provider.py``；
``pkgutil.iter_modules`` 会发现它，并在导入时触发 ``@register_provider``
装饰器。
"""

import importlib
import pkgutil

for _, module_name, _ in pkgutil.iter_modules(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)
