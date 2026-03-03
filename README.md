# 🤖 Jarvis — Agent Personnel Telegram

Agent IA personnel piloté via Telegram, avec mémoire persistante, scraping web et boucle de réflexion proactive.

## Fonctionnalités

- **Identité personnalisable** — Personnalité chargée depuis `soul.md`
- **Mémoire SQLite** — Historique des conversations + faits mémorisés (examens, deadlines…)
- **Scraping web** — Recherche DuckDuckGo + extraction de contenu (BeautifulSoup)
- **Proactivité** — Le LLM décide seul s'il a besoin de chercher sur le web avant de répondre
- **Interface Telegram** — Commandes `/start`, `/clear`, `/facts`, `/search`, `/forget`

## Structure

```
├── jarvis.py           # Point d'entrée
├── config.py           # Configuration (variables d'environnement)
├── brain.py            # Boucle de réflexion + appels LLM
├── memory.py           # Mémoire SQLite
├── scraper.py          # Scraping web
├── telegram_bot.py     # Interface Telegram
├── soul.md             # Personnalité de Jarvis
├── .env.example        # Modèle de fichier .env
├── Procfile            # Pour déploiement (Render, Railway…)
└── requirements.txt    # Dépendances Python
```

## Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/TON_USER/jarvis-bot.git
cd jarvis-bot

# 2. Créer un environnement virtuel
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer les secrets
copy .env.example .env    # Windows
# cp .env.example .env    # Linux/Mac
# Puis éditer .env avec tes vrais tokens

# 5. Lancer
python jarvis.py
```

## Variables d'environnement

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Token obtenu via [@BotFather](https://t.me/BotFather) |
| `OPENAI_API_KEY` | Clé API OpenAI |
| `ALLOWED_USER_IDS` | IDs Telegram autorisés, séparés par des virgules (optionnel) |
| `OPENAI_MODEL` | Modèle LLM (défaut : `gpt-4o-mini`) |

## Déploiement gratuit

Voir la section ci-dessous pour héberger Jarvis 24/7 gratuitement sur Render.

### Render.com (recommandé)

1. Push le repo sur GitHub
2. Va sur [render.com](https://render.com) → **New → Background Worker**
3. Connecte ton repo GitHub
4. Configure :
   - **Runtime** : Python
   - **Build Command** : `pip install -r requirements.txt`
   - **Start Command** : `python jarvis.py`
5. Ajoute les variables d'environnement dans **Environment**
6. Deploy !

## Licence

Projet personnel — usage privé.
