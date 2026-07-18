"""Web access capability provider — URL fetching and web search.

Provides up to two capabilities:
    * ``fetch_url`` — retrieve and extract visible text from a web page.
    * ``web_search`` — query a configured search API backend; it is not
      exposed to the model when no backend URL is configured.

Both enforce configurable timeouts and content-size caps.  JS-rendered
content is NOT supported (no headless browser).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

from agent_core.capability_registry import Capability
from agent_core.capabilities.base import (
    CapabilityProvider,
    ToolProviderConfigError,
    register_provider,
)


@dataclass(frozen=True)
class WebToolConfig:
    timeout_seconds: float = 15.0
    max_content_chars: int = 6000
    user_agent: str = "Mozilla/5.0 (compatible; LocalAgent/1.0)"
    search_api_url: Optional[str] = None
    search_api_key: Optional[str] = None


@register_provider
class WebCapabilityProvider(CapabilityProvider):
    """Web fetch + search.  TOML section: ``[tools.providers.web]``."""

    category = "web"

    def build(self, raw_config: dict[str, Any]) -> list[Capability]:
        try:
            config = WebToolConfig(**raw_config)
        except TypeError as exc:
            raise ToolProviderConfigError(
                f"web provider config has invalid fields: {exc}"
            ) from exc

        def _fetch_url(url: str) -> str:
            try:
                resp = requests.get(
                    url,
                    timeout=config.timeout_seconds,
                    headers={"User-Agent": config.user_agent},
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                return f"URL fetch failed: {exc}"

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())

            if len(text) > config.max_content_chars:
                text = (
                    text[: config.max_content_chars]
                    + f"\n…(truncated, {len(text)} chars total)"
                )
            return text or "(no extractable text on page)"

        def _web_search(query: str) -> str:
            if not config.search_api_url:
                return (
                    "Web search is not configured (missing search_api_url). "
                    "Contact the administrator to set up a search backend."
                )
            try:
                resp = requests.get(
                    config.search_api_url,
                    params={"q": query},
                    headers=(
                        {"Authorization": f"Bearer {config.search_api_key}"}
                        if config.search_api_key
                        else {}
                    ),
                    timeout=config.timeout_seconds,
                )
                resp.raise_for_status()
                data = resp.json()
            except (requests.RequestException, ValueError) as exc:
                return f"Search failed: {exc}"

            lines: list[str] = []
            for item in data.get("results", [])[:5]:
                lines.append(
                    f"- {item.get('title', '?')}: {item.get('snippet', '?')} "
                    f"({item.get('url', '?')})"
                )
            return "\n".join(lines) if lines else "No results found."

        capabilities = [
            Capability(
                name="fetch_url",
                description=(
                    "Fetch a web page at the given URL and extract its visible "
                    "text content (scripts and styles are stripped)."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "Full URL including http(s):// prefix",
                        }
                    },
                    "required": ["url"],
                },
                handler=_fetch_url,
            )
        ]
        if config.search_api_url:
            capabilities.append(
                Capability(
                    name="web_search",
                    description=(
                        "Search the web for the given query and return "
                        "title + snippet + URL results."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search keywords",
                            }
                        },
                        "required": ["query"],
                    },
                    handler=_web_search,
                )
            )
        return capabilities
