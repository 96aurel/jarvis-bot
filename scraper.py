"""
scraper.py — Outil de scraping web pour Jarvis.

Deux modes :
  1. Recherche Google (via scraping de la page de résultats)
  2. Extraction du contenu d'une URL spécifique

Utilise requests + BeautifulSoup (léger, pas besoin de navigateur).
"""

import re
import logging
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger("jarvis.scraper")

_session = requests.Session()
_session.headers.update({
    "User-Agent": config.USER_AGENT,
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
})


# ── Recherche web ────────────────────────────────────────

def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Effectue une recherche web via DuckDuckGo HTML (pas d'API key requise).
    Retourne une liste de dicts : {"title", "url", "snippet"}.
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Erreur lors de la recherche web : %s", e)
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for item in soup.select(".result__body")[:num_results]:
        title_tag = item.select_one(".result__title a")
        snippet_tag = item.select_one(".result__snippet")

        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        link = title_tag.get("href", "")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

        # DuckDuckGo utilise parfois des redirections
        if "uddg=" in link:
            from urllib.parse import parse_qs, urlparse
            parsed = urlparse(link)
            qs = parse_qs(parsed.query)
            link = qs.get("uddg", [link])[0]

        results.append({"title": title, "url": link, "snippet": snippet})

    logger.info("Recherche '%s' → %d résultats", query, len(results))
    return results


# ── Extraction de contenu ────────────────────────────────

def scrape_url(url: str, max_chars: int | None = None) -> str:
    """
    Télécharge une page et extrait le texte principal.
    Retourne le contenu textuel nettoyé.
    """
    max_chars = max_chars or config.SCRAPE_MAX_CHARS
    try:
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Erreur scraping %s : %s", url, e)
        return f"[Erreur : impossible d'accéder à {url}]"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Supprime les éléments non pertinents
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    # Essaie de trouver le contenu principal
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.find("div", id=re.compile(r"content|main", re.I))
        or soup.body
    )

    if not main:
        return "[Erreur : aucun contenu trouvé sur la page]"

    text = main.get_text(separator="\n", strip=True)

    # Nettoyage : supprime les lignes vides multiples
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[… contenu tronqué]"

    return text


# ── Recherche + extraction combinée ─────────────────────

def search_and_summarize(query: str) -> str:
    """
    Recherche sur le web et extrait le contenu de la première page pertinente.
    Retourne un texte prêt à être injecté dans le prompt du LLM.
    """
    results = web_search(query, num_results=3)
    if not results:
        return "Aucun résultat trouvé pour cette recherche."

    output_parts = [f"**Résultats pour : « {query} »**\n"]

    for i, r in enumerate(results, 1):
        output_parts.append(f"{i}. [{r['title']}]({r['url']})")
        output_parts.append(f"   {r['snippet']}\n")

    # Extraire le contenu du premier résultat
    if results[0]["url"]:
        content = scrape_url(results[0]["url"], max_chars=2000)
        output_parts.append(f"\n**Extrait du premier résultat :**\n{content}")

    return "\n".join(output_parts)
