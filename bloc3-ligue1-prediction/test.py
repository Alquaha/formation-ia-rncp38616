import sqlite3
import pandas as pd

# ==========================
# Connexion à la base
# ==========================

db_path = "database.sqlite"
conn = sqlite3.connect(db_path)

print("\nConnexion réussie à la base SQLite")

# ==========================
# Afficher les tables
# ==========================

tables = pd.read_sql("""
SELECT name
FROM sqlite_master
WHERE type='table';
""", conn)

print("\nTables disponibles :")
print(tables.to_string(index=False))

# ==========================
# Voir les colonnes de Match
# ==========================

columns = pd.read_sql("""
PRAGMA table_info(Match);
""", conn)

print("\nColonnes de la table Match :")
print(columns[['name', 'type']].to_string(index=False))

# ==========================
# Aperçu des données Match
# ==========================

sample = pd.read_sql("""
SELECT
    date,
    home_team_api_id,
    away_team_api_id,
    home_team_goal,
    away_team_goal
FROM Match
LIMIT 10;
""", conn)

print("\nExemple de matchs :")
print(sample.to_string(index=False))

# ==========================
# Nombre total de matchs
# ==========================

count = pd.read_sql("""
SELECT COUNT(*) as total_matches
FROM Match;
""", conn)

print("\nNombre total de matchs dans la base :")
print(count.to_string(index=False))

# ==========================
# Fermeture connexion
# ==========================

conn.close()

print("\nConnexion fermée")