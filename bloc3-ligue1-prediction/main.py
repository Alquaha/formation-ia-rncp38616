import sqlite3
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping

# ==========================
# 1️⃣ Connexion & extraction Ligue 1
# ==========================

db_path = "database.sqlite"
conn = sqlite3.connect(db_path)

ligue1_id = 4769

ligue1_matches = pd.read_sql(f"""
SELECT *
FROM Match
WHERE league_id = {ligue1_id}
""", conn)

conn.close()

print("Nombre de matchs Ligue 1 :", ligue1_matches.shape[0])

# ==========================
# 2️⃣ Création variable cible
# ==========================

def get_result(row):
    if row['home_team_goal'] > row['away_team_goal']:
        return 2
    elif row['home_team_goal'] < row['away_team_goal']:
        return 0
    else:
        return 1

ligue1_matches['FTR'] = ligue1_matches.apply(get_result, axis=1)

print("Distribution des classes :")
print(ligue1_matches['FTR'].value_counts())

# ==========================
# 3️⃣ Tri par date
# ==========================

ligue1_matches['date'] = pd.to_datetime(ligue1_matches['date'])
ligue1_matches = ligue1_matches.sort_values('date')
ligue1_matches = ligue1_matches.reset_index(drop=True)

print("Dataset trié par date ✅")

# ==========================
# 4️⃣ Moyenne buts 5 derniers matchs
# ==========================

ligue1_matches['home_avg_goals_last5'] = 0.0
ligue1_matches['away_avg_goals_last5'] = 0.0

team_history = {}

for i, row in ligue1_matches.iterrows():
    home = row['home_team_api_id']
    away = row['away_team_api_id']

    if home not in team_history:
        team_history[home] = []
    if away not in team_history:
        team_history[away] = []

    if len(team_history[home]) > 0:
        ligue1_matches.at[i, 'home_avg_goals_last5'] = \
            sum(team_history[home][-5:]) / len(team_history[home][-5:])

    if len(team_history[away]) > 0:
        ligue1_matches.at[i, 'away_avg_goals_last5'] = \
            sum(team_history[away][-5:]) / len(team_history[away][-5:])

    team_history[home].append(row['home_team_goal'])
    team_history[away].append(row['away_team_goal'])

print("Feature offensive créée ✅")

# ==========================
# 5️⃣ Moyenne buts encaissés
# ==========================

ligue1_matches['home_avg_conceded_last5'] = 0.0
ligue1_matches['away_avg_conceded_last5'] = 0.0

team_conceded_history = {}

for i, row in ligue1_matches.iterrows():
    home = row['home_team_api_id']
    away = row['away_team_api_id']

    if home not in team_conceded_history:
        team_conceded_history[home] = []
    if away not in team_conceded_history:
        team_conceded_history[away] = []

    if len(team_conceded_history[home]) > 0:
        ligue1_matches.at[i, 'home_avg_conceded_last5'] = \
            sum(team_conceded_history[home][-5:]) / len(team_conceded_history[home][-5:])

    if len(team_conceded_history[away]) > 0:
        ligue1_matches.at[i, 'away_avg_conceded_last5'] = \
            sum(team_conceded_history[away][-5:]) / len(team_conceded_history[away][-5:])

    team_conceded_history[home].append(row['away_team_goal'])
    team_conceded_history[away].append(row['home_team_goal'])

print("Feature défensive créée ✅")

# ==========================
# 6️⃣ Différences attaque / défense
# ==========================

ligue1_matches['attack_diff'] = (
    ligue1_matches['home_avg_goals_last5'] -
    ligue1_matches['away_avg_goals_last5']
)

ligue1_matches['defense_diff'] = (
    ligue1_matches['away_avg_conceded_last5'] -
    ligue1_matches['home_avg_conceded_last5']
)

print("Features différentielles créées ")

# ==========================
# 7️⃣ Points dynamiques
# ==========================

ligue1_matches['home_points_last5'] = 0.0
ligue1_matches['away_points_last5'] = 0.0

team_points_history = {}

for i, row in ligue1_matches.iterrows():
    home = row['home_team_api_id']
    away = row['away_team_api_id']

    if home not in team_points_history:
        team_points_history[home] = []
    if away not in team_points_history:
        team_points_history[away] = []

    if len(team_points_history[home]) > 0:
        ligue1_matches.at[i, 'home_points_last5'] = \
            sum(team_points_history[home][-5:]) / len(team_points_history[home][-5:])

    if len(team_points_history[away]) > 0:
        ligue1_matches.at[i, 'away_points_last5'] = \
            sum(team_points_history[away][-5:]) / len(team_points_history[away][-5:])

    if row['FTR'] == 2:
        team_points_history[home].append(3)
        team_points_history[away].append(0)
    elif row['FTR'] == 0:
        team_points_history[home].append(0)
        team_points_history[away].append(3)
    else:
        team_points_history[home].append(1)
        team_points_history[away].append(1)

ligue1_matches['points_diff'] = (
    ligue1_matches['home_points_last5'] -
    ligue1_matches['away_points_last5']
)

print("Features points dynamiques créées ✅")

# ==========================
# 8️⃣ Performance domicile / extérieur
# ==========================

ligue1_matches['home_home_points_last5'] = 0.0
ligue1_matches['away_away_points_last5'] = 0.0

home_special_history = {}
away_special_history = {}

for i, row in ligue1_matches.iterrows():
    home = row['home_team_api_id']
    away = row['away_team_api_id']

    if home not in home_special_history:
        home_special_history[home] = []
    if away not in away_special_history:
        away_special_history[away] = []

    if len(home_special_history[home]) > 0:
        ligue1_matches.at[i, 'home_home_points_last5'] = \
            sum(home_special_history[home][-5:]) / len(home_special_history[home][-5:])

    if len(away_special_history[away]) > 0:
        ligue1_matches.at[i, 'away_away_points_last5'] = \
            sum(away_special_history[away][-5:]) / len(away_special_history[away][-5:])

    if row['FTR'] == 2:
        home_special_history[home].append(3)
        away_special_history[away].append(0)
    elif row['FTR'] == 0:
        home_special_history[home].append(0)
        away_special_history[away].append(3)
    else:
        home_special_history[home].append(1)
        away_special_history[away].append(1)

ligue1_matches['home_special_diff'] = (
    ligue1_matches['home_home_points_last5'] -
    ligue1_matches['away_away_points_last5']
)

print("Features domicile/extérieur créées ✅")

# ==========================
# 9️⃣  ELO avec avantage domicile
# ==========================

INITIAL_ELO = 1500
K = 20
HOME_ADVANTAGE = 100

ligue1_matches['home_elo'] = 0.0
ligue1_matches['away_elo'] = 0.0
ligue1_matches['elo_diff'] = 0.0

elo_dict = {}

for i, row in ligue1_matches.iterrows():

    home = row['home_team_api_id']
    away = row['away_team_api_id']

    if home not in elo_dict:
        elo_dict[home] = INITIAL_ELO
    if away not in elo_dict:
        elo_dict[away] = INITIAL_ELO

    home_elo = elo_dict[home]
    away_elo = elo_dict[away]

    ligue1_matches.at[i, 'home_elo'] = home_elo
    ligue1_matches.at[i, 'away_elo'] = away_elo
    ligue1_matches.at[i, 'elo_diff'] = home_elo - away_elo

    expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + HOME_ADVANTAGE)) / 400))

    if row['FTR'] == 2:
        actual_home = 1
    elif row['FTR'] == 1:
        actual_home = 0.5
    else:
        actual_home = 0

    elo_dict[home] = home_elo + K * (actual_home - expected_home)
    elo_dict[away] = away_elo + K * ((1 - actual_home) - (1 - expected_home))

print("ELO rating créé ✅")

# ==========================
# 🔟 Nettoyage
# ==========================

ligue1_matches = ligue1_matches.iloc[100:]

# ==========================
# 1️⃣1️⃣ Dataset ML
# ==========================

features = [
    'attack_diff',
    'defense_diff',
    'points_diff',
    'home_special_diff',
    'elo_diff'
]

X = ligue1_matches[features]
y = ligue1_matches['FTR']

train_size = int(len(X) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]
y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

print("Train size :", len(X_train))
print("Test size :", len(X_test))

# ==========================
# 1️⃣2️⃣ XGBoost Tuné
# ==========================

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=20,
    random_state=999
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)

print("Best iteration :", xgb_model.best_iteration)
xgb_preds = xgb_model.predict(X_test)
print("\n=== XGBoost FINAL AVEC ELO + Early Stopping ===")
print("Accuracy :", accuracy_score(y_test, xgb_preds))
print(classification_report(y_test, xgb_preds))


def predict_match(home_id, away_id):

    # Récupérer dernières lignes pour chaque équipe
    home_data = ligue1_matches[
        ligue1_matches['home_team_api_id'] == home_id
    ].iloc[-1]

    away_data = ligue1_matches[
        ligue1_matches['away_team_api_id'] == away_id
    ].iloc[-1]

    attack_diff = home_data['home_avg_goals_last5'] - away_data['away_avg_goals_last5']
    defense_diff = away_data['away_avg_conceded_last5'] - home_data['home_avg_conceded_last5']
    points_diff = home_data['home_points_last5'] - away_data['away_points_last5']
    home_special_diff = home_data['home_home_points_last5'] - away_data['away_away_points_last5']
    elo_diff = home_data['home_elo'] - away_data['away_elo']

    new_match = pd.DataFrame([[
        attack_diff,
        defense_diff,
        points_diff,
        home_special_diff,
        elo_diff
    ]], columns=features)

    probs = xgb_model.predict_proba(new_match)[0]

    print("\n=== PRÉDICTION MATCH ===")
    print("Victoire Extérieur :", round(probs[0] * 100, 2), "%")
    print("Match Nul :", round(probs[1] * 100, 2), "%")
    print("Victoire Domicile :", round(probs[2] * 100, 2), "%")

    # ==========================
# Test prédiction
# ==========================

predict_match(9827, 9853)