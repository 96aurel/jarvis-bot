"""
scraper.py — Outil de scraping web pour Jarvis.

Deux modes :
  1. Recherche web (via duckduckgo-search ou fallback HTML)
  2. Extraction du contenu d'une URL spécifique

Utilise requests + BeautifulSoup (léger, pas besoin de navigateur).
"""

import re
import time
import logging
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

import config

logger = logging.getLogger("jarvis.scraper")

_session = requests.Session()
_session.headers.update({
    "User-Agent": config.USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
})


# ── Recherche web ────────────────────────────────────────

def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Recherche web. Essaie d'abord duckduckgo-search (fiable),
    puis fallback sur scraping HTML si la lib n'est pas dispo.
    """
    # Methode 1 : duckduckgo-search (gere l'anti-bot)
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        if results:
            logger.info("Recherche DDG lib '%s' → %d resultats", query, len(results))
            return results
    except ImportError:
        logger.debug("duckduckgo-search non installe, fallback HTML")
    except Exception as e:
        logger.warning("DDG lib echouee : %s — fallback HTML", str(e)[:150])

    # Methode 2 : scraping HTML DuckDuckGo (fallback)
    return _web_search_html(query, num_results)


def _web_search_html(query: str, num_results: int = 5) -> list[dict]:
    """Fallback : scraping HTML de DuckDuckGo."""
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    for attempt in range(2):
        try:
            resp = _session.get(url, timeout=12)
            resp.raise_for_status()

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

                if "uddg=" in link:
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(link)
                    qs = parse_qs(parsed.query)
                    link = qs.get("uddg", [link])[0]

                results.append({"title": title, "url": link, "snippet": snippet})

            if results:
                logger.info("Recherche HTML '%s' → %d resultats", query, len(results))
                return results

            logger.warning("Recherche HTML vide (tentative %d)", attempt + 1)
            time.sleep(1.5)

        except requests.RequestException as e:
            logger.error("Erreur recherche HTML (tentative %d) : %s", attempt + 1, e)
            time.sleep(1.5)

    return []


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
    Recherche sur le web et extrait le contenu des pages pertinentes.
    Retourne un texte pret a etre injecte dans le prompt du LLM.
    """
    results = web_search(query, num_results=5)
    if not results:
        return f"[RECHERCHE ECHOUEE] Aucun resultat trouve pour : {query}"

    output_parts = [f"Resultats de recherche pour : {query}\n"]

    for i, r in enumerate(results, 1):
        output_parts.append(f"{i}. {r['title']}")
        output_parts.append(f"   URL: {r['url']}")
        if r['snippet']:
            output_parts.append(f"   Resume: {r['snippet']}")
        output_parts.append("")

    # Extraire le contenu des 2 premieres pages (pas juste la premiere)
    extracted = 0
    for r in results[:3]:
        url = r.get("url", "")
        if not url or url.startswith("#"):
            continue
        try:
            content = scrape_url(url, max_chars=2500)
            if content and not content.startswith("[Erreur"):
                output_parts.append(f"--- Contenu de : {r['title']} ---")
                output_parts.append(content)
                output_parts.append("")
                extracted += 1
                if extracted >= 2:
                    break
        except Exception as e:
            logger.warning("Echec extraction %s : %s", url, e)

    if extracted == 0:
        output_parts.append("[Aucun contenu de page n'a pu etre extrait, seuls les snippets sont disponibles]")

    return "\n".join(output_parts)
