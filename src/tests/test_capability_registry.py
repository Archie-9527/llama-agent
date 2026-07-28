"""capability_registry.py 测试。

覆盖范围：
  - @register 装饰器创建 Capability 条目。
  - get_capability() 按名称查找。
  - list_capabilities() 返回快照。
  - 重复注册抛出 ValueError。
  - 能力缺失时抛出 KeyError。
  - 使用 clear_registry() 隔离测试。
  - Capability DataClass 的不可变性。
"""

from __future__ import annotations

import os
import sys

import pytest

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src not in sys.path:
    sys.path.insert(0, _src)

from agent_core.capability_registry import (
    Capability,
    clear_registry,
    get_capability,
    list_capabilities,
    register,
)


@pytest.fixture(autouse=True)
def _reset_registry():
    """在每个测试前后清空注册表。"""
    clear_registry()
    yield
    clear_registry()


class TestRegister:
    """测试基于装饰器的注册。"""

    def test_register_creates_capability(self):
        @register(
            name="test_tool",
            description="A test tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
        )
        def test_tool(x: int) -> int:
            return x * 2

        cap = get_capability("test_tool")
        assert cap.name == "test_tool"
        assert cap.description == "A test tool"
        assert cap.input_schema["required"] == ["x"]
        assert cap.handler(5) == 10  # Handler 就是原始函数

    def test_duplicate_name_raises(self):
        @register(
            name="dup",
            description="First",
            input_schema={"type": "object"},
        )
        def first() -> str:
            return "a"

        with pytest.raises(ValueError, match="already registered"):
            @register(
                name="dup",
                description="Second",
                input_schema={"type": "object"},
            )
            def second() -> str:
                return "b"

    def test_multiple_registrations(self):
        @register(name="a", description="A", input_schema={"type": "object"})
        def fn_a() -> str:
            return "a"

        @register(name="b", description="B", input_schema={"type": "object"})
        def fn_b() -> str:
            return "b"

        caps = list_capabilities()
        names = {c.name for c in caps}
        assert names == {"a", "b"}


class TestGetCapability:
    """测试按名称查找。"""

    def test_get_existing(self):
        @register(name="find_me", description="x", input_schema={})
        def find_me() -> str:
            return "found"

        cap = get_capability("find_me")
        assert cap.name == "find_me"

    def test_get_missing_raises_keyerror(self):
        with pytest.raises(KeyError, match="missing_tool"):
            get_capability("missing_tool")


class TestListCapabilities:
    """测试快照语义。"""

    def test_list_empty(self):
        assert list_capabilities() == []

    def test_list_returns_copy(self):
        @register(name="t1", description="", input_schema={})
        def t1() -> None: ...

        caps = list_capabilities()
        caps.clear()  # 不应影响真实注册表
        assert len(list_capabilities()) == 1

    def test_list_includes_all_registered(self):
        @register(name="x1", description="", input_schema={})
        def x1() -> None: ...

        @register(name="x2", description="", input_schema={})
        def x2() -> None: ...

        names = {c.name for c in list_capabilities()}
        assert names == {"x1", "x2"}


class TestCapabilityDataclass:
    """测试不可变性。"""

    def test_frozen(self):
        cap = Capability(
            name="frozen",
            description="immutable",
            handler=lambda: None,
            input_schema={},
        )
        with pytest.raises(Exception):
            cap.name = "changed"  # type: ignore[misc]

    def test_repr(self):
        cap = Capability(name="r", description="d", handler=lambda: 1, input_schema={})
        rep = repr(cap)
        assert "r" in rep
