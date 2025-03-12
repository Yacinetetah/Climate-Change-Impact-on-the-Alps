# Le changement climatique peut il être observé dans les Alpes?

## Consignes

Ce TP peut être réalisé sur [MyDocker]. XX TODO ADD
Vous devrez rendre ce notebook avec les fonctions/ votre script python ainsi qu'un compte-rendu de 8 pages maximum dans lequel vous répondrez aux questions.
**Le compte-rendu au format PDF et le notebook** sont à rendre par email avant le prochain cours soit **lundi 17 janvier 23h59**. J'ouvrirai mes mails au réveil, et à partir de là, -2 points par heure de retard.

## Introduction

En science des données vous devrez préparer les données, les analyser (statistiquement) et produire des figures pertinentes dans l'objectif de répondre à différentes questions.

Dans ce TP, on se demande si le changement climatique est visible dans les Alpes et nous mettrons en lien les observations avec les analyses des flux hydrométriques (= le force du courant dans les rivières) dans cette région. 

Pour ce faire nous allons commencer par étudier l'évolution de la météo au cours des dernières décennies. Météo France, l'organisme national de météorologie en France, a déposé des données climatologiques par département avec de nombreux paramètres disponibles sur le site [data.gouv](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-quotidiennes/).

Dans un second temps, nous étudierons l'évolution des débits de l'Arve (une rivière) en lien avec le changement climatique et la fonte des glaciers aux alentours du Mont-Blanc (un massif *relativement* connu des Alpes). Le papier a été publié dans Scientific Reports en 2020 et est disponible [ici](https://doi.org/10.1038/s41598-020-67379-7).

## Chargement des librairies


Voici quelques librairies dont vous aurez très probablement besoin, n'hésitez pas a en ajouter d'autres !

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

```

## Optimisation du code

Pensez à optimiser votre code pour améliorer l'efficacité, en particulier lorsqu'il s'agit de manipuler de grandes quantités de données. Utilisez des fonctions pour que vous puissiez refaire la même analyse avec un autre département.


# PARTIE 1: Données climatiques de Météo France

## Chargement du jeu de données 

Commencez par récupérer sur le site des données publiques francaises, les données quotidiennes de météo entre 1950 et 2022 en Haute-Savoie où se situe le Mont-Blanc (département n°74) sur le site [data.gouv](https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-quotidiennes/).

On ne s'intéresse pour le moment qu'aux données de température et de précipitation.

Quand vous êtes prête ou prêt, chargez la table d'intéret dans la variable `meteo`.

*Indication :* Pour trouver quel fichier télécharger, lisez la documentation!

```python
#Votre code ici
meteo = pd.read_csv("Q_74_previous-1950-2023_RR-T-Vent.csv", sep=';')  
```

## Exploration des données

Décrivez la table. Par exempl,e répondez aux questions suivantes:

- Quelle sont les dimensions de `meteo` ?
- Combien y a-t-il de postes météorologiques ? 
    
Vos réponses sont à rédiger dans le compte-rendu pas dans le notebook, ici je n'évaluerai que les fonctions!

```python
#Votre code ici
dimension=meteo.shape
print(dimension)
#Votre code ici
nombre_de_postes = meteo['NUM_POSTE'].nunique()
print(f"Le nombre de postes distincts est : {nombre_de_postes}")
```

**Gestion des valeurs manquantes et filtration**

- Quelles colonnes allez vous sélectionner pour notre étude ? Pour rappel, on va étudier l'évolution de la température et des précipitations depuis les années 1950 dans les Alpes.
*Indication:* La localisation des postes pourra être utile.
- Créer la table `meteo_cleaned` avec les colonnes d'interet et sans données manquantes.
- Faites une première analyse sur les informations statistiques de base sur les données (moyenne, médiane, écart-type, etc.).


```python
# === 1) Renommer les colonnes ===
meteo = meteo.rename(columns={
    "AAAAMMJJ": "Date",
    "RR": "Précipitations",
    "TM": "Température_moyenne",
    "TX": "Température_maximale",
    "TN": "Température_minimale",
    "LAT": "Latitude",
    "LON": "Longitude",
    "ALTI": "Altitude",
    "NOM_USUEL": "Nom_poste",
    "NUM_POSTE": "NUM_POSTE"  # inchangé, mais rappel
})

# === 2) Choix des colonnes d’intérêt ===
colonnes_interet = [
    "Date",
    "Précipitations",
    "Température_moyenne",
    "Température_maximale",
    "Température_minimale",
    "Latitude",
    "Longitude",
    "Altitude",
    "Nom_poste",
    "NUM_POSTE"
]

# On réduit le DataFrame aux seules colonnes utiles
meteo = meteo[colonnes_interet]

# === 3) (Optionnel) Remplacement / Imputation de valeurs manquantes ===
# Par exemple, si vous voulez remplacer les NaN de "Précipitations" par 0 :
# meteo["Précipitations"] = meteo["Précipitations"].fillna(0)
# meteo["Température_moyenne"] = meteo["Température_moyenne"].fillna(meteo["Température_moyenne"].mean())
# etc. selon vos besoins

# === 4) Suppression des lignes restantes qui contiennent des NaN ===
meteo_cleaned = meteo.dropna()

# === 5) Vérification : affichage des premières lignes ===
print("=== Aperçu des données nettoyées ===")
print(meteo_cleaned.head())

# === 6) Analyse statistique de base ===
stats = meteo_cleaned[[
    "Précipitations",
    "Température_moyenne",
    "Température_maximale",
    "Température_minimale"
]].describe()

print("\n=== Analyse statistique ===")
print(stats)
meteo_cleaned.describe()
```

- Combien de stations restent dans notre étude ? Où se situent-elles ?

```python
# === 7) Combien de stations restent dans l’étude ? ===
stations_restantes = meteo_cleaned["NUM_POSTE"].nunique()
print(f"\nNombre de stations restantes dans l'étude : {stations_restantes}")

# === 8) Où se situent-elles ? Liste des stations uniques ===
stations_localisation = (
    meteo_cleaned[["NUM_POSTE", "Nom_poste", "Latitude", "Longitude", "Altitude"]]
    .drop_duplicates(subset="NUM_POSTE")
)

print("\nInformations sur les stations (une ligne par station) :")
print(stations_localisation)

# Facultatif : Tracer l’étendue géographique (lat/lon min/max)
lat_min, lat_max = stations_localisation["Latitude"].min(), stations_localisation["Latitude"].max()
lon_min, lon_max = stations_localisation["Longitude"].min(), stations_localisation["Longitude"].max()
alti_min, alti_max = stations_localisation["Altitude"].min(), stations_localisation["Altitude"].max()

print(f"\nÉtendue des latitudes : {lat_min}° à {lat_max}°")
print(f"Étendue des longitudes : {lon_min}° à {lon_max}°")
print(f"Étendue des altitudes : {alti_min} m à {alti_max} m")
```

## Analyse des données
### Tendances annuelles

Quelles sont les tendances annuelles dans les données météorologiques depuis 1950 ? La température moyenne a-t-elle changée ? Est-ce qu'il y a plus ou moins de précipitations ?

Faites une analyse (calcul de moyenne, tests de regressions etc) pour répondre à la question. Ci-dessous voici les principales étapes à effectuer:
- Transformez la colonne date pour que vous puissiez l'exploiter facilement
- Calculez les températures et précipitations moyennes annuelles
- Faites une régression linéaire pour estimer l'évolution de ces valeurs. Utilisez scikit-learn comme appris en L1 ISD.
- Evaluez la performance de vos modèles de prédiction. Pour cela, vous pourrez par exemple utiliser les métriques disponibles dans `sklearn.metrics`.
- La régression est-elle pertinente pour la température ? Pour les précipitations ?
- Auriez-vous un autre modèle plus pertinent qu'une régression linéaire à proposer ? N'hésitez pas à l'**ajouter**.
- Crééz également un ou plusieurs graphiques pour représenter les variations des paramètres météorologiques au fil du temps.

Si besoin vous pouvez charger de nouvelles librairies.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
# ===============================
# 1) Copie du DataFrame nettoyé
# ===============================
meteo_cleaned = meteo_cleaned.copy()

# ===============================
# 2) Conversion du champ "Date"
#    au format datetime
# ===============================
meteo_cleaned["Date"] = pd.to_datetime(
    meteo_cleaned["Date"],   # votre colonne date (ex: AAAAMMJJ)
    format="%Y%m%d",         # correspond à l'année, mois, jour
    errors="coerce"          # force à NaT si erreur
)

# ===============================
# 3) Extraction de l'année
# ===============================
meteo_cleaned["Annee"] = meteo_cleaned["Date"].dt.year

print("Liste des années détectées :")
print(meteo_cleaned["Annee"].unique())

moyennes_annuelles = (
    meteo_cleaned
    .groupby("Annee")[["Température_moyenne", "Précipitations"]]
    .mean()
    .reset_index()
)

print("=== Moyennes annuelles ===")
print(moyennes_annuelles.head())

# ===============================
meteo = meteo[colonnes_interet].dropna().copy()
meteo["Date"] = pd.to_datetime(meteo["Date"], format="%Y%m%d")
meteo["Année"] = meteo["Date"].dt.year

# Moyennes annuelles
moyennes_annuelles = meteo.groupby("Année")[
    ["Température_moyenne", "Précipitations"]
].mean().reset_index()

# Modèle de régression linéaire
X = moyennes_annuelles["Année"].values.reshape(-1, 1)
y_temp = moyennes_annuelles["Température_moyenne"].values
y_precip = moyennes_annuelles["Précipitations"].values

model_temp = LinearRegression()
model_temp.fit(X, y_temp)

model_precip = LinearRegression()
model_precip.fit(X, y_precip)

# Modèle alternatif : Random Forest
rf_temp = RandomForestRegressor(random_state=42)
rf_temp.fit(X, y_temp)

rf_precip = RandomForestRegressor(random_state=42)
rf_precip.fit(X, y_precip)

# Prédictions
# Prédictions
annee_cible = np.array([[2100]])
prediction_lin = model_temp.predict(annee_cible)[0]
prediction_rf = rf_temp.predict(annee_cible)[0]

print(f"Prédiction Température en 2100 avec Régression Linéaire : {prediction_lin:.2f} °C")
print(f"Prédiction Température en 2100 avec Random Forest : {prediction_rf:.2f} °C")

# Graphiques
plt.figure(figsize=(12, 6))
plt.plot(moyennes_annuelles["Année"], y_temp, label="Température Moyenne", marker="o")
plt.plot(moyennes_annuelles["Année"], model_temp.predict(X), label="Régression Linéaire", linestyle="--")
plt.plot(moyennes_annuelles["Année"], rf_temp.predict(X), label="Random Forest", linestyle="-")
plt.title("Évolution de la Température Moyenne")
plt.xlabel("Année")
plt.ylabel("Température (°C)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(moyennes_annuelles["Année"], y_precip, label="Précipitations Moyennes", marker="o")
plt.plot(moyennes_annuelles["Année"], model_precip.predict(X), label="Régression Linéaire", linestyle="--")
plt.plot(moyennes_annuelles["Année"], rf_precip.predict(X), label="Random Forest", linestyle="-")
plt.title("Évolution des Précipitations")
plt.xlabel("Année")
plt.ylabel("Précipitations (mm)")
plt.legend()
plt.grid()
plt.show()

# Visualisation des tendances
plt.figure(figsize=(14, 6))
plt.plot(moyennes_annuelles["Année"], y_temp, label="Température Moyenne", color="blue")
plt.plot(moyennes_annuelles["Année"], y_precip, label="Précipitations Moyennes", color="green")
plt.title("Tendances des Paramètres Météorologiques")
plt.xlabel("Année")
plt.ylabel("Valeur")
plt.legend()
plt.grid()
plt.show()

# Visualisation des boxplots
meteo["Décennie"] = meteo["Année"] // 10 * 10
plt.figure(figsize=(12, 6))
sns.boxplot(x="Décennie", y="Température_moyenne", data=meteo)
plt.title("Distribution de la Température Moyenne par décennie")
plt.xlabel("Décennie")
plt.ylabel("Température Moyenne (°C)")
plt.grid()
plt.show()

# Comptage des années chaudes
meteo["is_hot"] = meteo["Température_maximale"] > 28
annees_chaudes = meteo.groupby("Année")["is_hot"].sum()
plt.figure(figsize=(10, 5))
plt.plot(annees_chaudes)
plt.title("Jours chauds par année")
plt.xlabel("Année")
plt.ylabel("Nombre de jours chauds")
plt.grid()
plt.show()



```

### Prévisions

Quelle température fera-t-il en 2100 selon votre modèle ?


```python
# 9) Exemple: prédire pour l'année 2100
# ===============================
annee_cible = np.array([[2100]])
temp_2100 = model_temp.predict(annee_cible)
precip_2100 = model_precip.predict(annee_cible)

print(f"\nTempérature prédite en 2100 : {temp_2100[0]:.2f} °C")
print(f"Précipitations prédites en 2100 : {precip_2100[0]:.2f}")
```

### Variabilité saisonnière 

Analysez la variabilité saisonnière des données. 

- Manipulez les données pour les regrouper mois par mois. Y a-t-il des tendances ? Correspondent-elles à vos connaissances ?
- N'oubliez pas de visualisez les données
- Enfin si vous êtes à l'aise, étudiez en plus les épisodes de fortes chaleurs (températures > 28°C). Est-ce qu'il y en a plus souvent plus récemment?



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Forcer la copie de meteo_cleaned au tout début
meteo_cleaned = meteo_cleaned.copy()

# 2) Conversion en datetime et extraction Année, Mois via .assign()
#    Cela crée un nouveau DataFrame et évite le SettingWithCopyWarning
meteo_cleaned = (
    meteo_cleaned
    .assign(
        Date=lambda df: pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce"),
        Année=lambda df: df["Date"].dt.year,
        month=lambda df: df["Date"].dt.month
    )
)

# 3) Calcul mensuel
monthly_data = (
    meteo_cleaned
    .groupby(["Année", "month"], as_index=False)
    .agg({
        "Température_moyenne": "mean",
        "Précipitations": "sum"
    })
)

print("=== Aperçu mensuel ===")
print(monthly_data.head())

# 4) Visualisation de la température mensuelle
plt.figure(figsize=(12,5))
dates_mensuelles = pd.to_datetime(
    monthly_data["Année"].astype(str) + "-" + monthly_data["month"].astype(str)
)
plt.plot(dates_mensuelles, monthly_data["Température_moyenne"], marker='o')
plt.title("Température moyenne mensuelle au fil du temps")
plt.xlabel("Date (Année-Mois)")
plt.ylabel("Température (°C)")
plt.grid(True)
plt.show()

# 5) Boxplots par mois : on re-copie meteo_cleaned pour éviter tout avertissement
df_boxplot = meteo_cleaned.copy()

plt.figure(figsize=(10,6))
sns.boxplot(x="month", y="Température_moyenne", data=df_boxplot)
plt.title("Distribution de la Température Moyenne par mois")
plt.show()

# 6) Définir les saisons en utilisant .assign + une fonction
def label_saison(mois):
    if mois in [12, 1, 2]:
        return "Hiver"
    elif mois in [3, 4, 5]:
        return "Printemps"
    elif mois in [6, 7, 8]:
        return "Été"
    else:
        return "Automne"

meteo_cleaned = (
    meteo_cleaned
    .assign(Saison=lambda df: df["month"].apply(label_saison))
)

# 7) Déterminer les jours chauds (Tmax > 28 °C), toujours via .assign()
meteo_cleaned = (
    meteo_cleaned
    .assign(is_hot_day=lambda df: (df["Température_maximale"] > 28).astype(int))
)

# 8) Comptage des jours chauds par année
heatwaves_annual = (
    meteo_cleaned
    .groupby("Année", as_index=False)["is_hot_day"]
    .sum()
    .rename(columns={"is_hot_day": "Nb_jours_chauds"})
)

print("\n=== Nombre de jours chauds par an (dern. lignes) ===")
print(heatwaves_annual.tail())

plt.figure(figsize=(8,5))
plt.plot(heatwaves_annual["Année"], heatwaves_annual["Nb_jours_chauds"], marker='o', color="red")
plt.title("Évolution du nombre de jours chauds (> 28°C) par an")
plt.xlabel("Année")
plt.ylabel("Nombre de jours chauds")
plt.grid(True)
plt.show()
```

<!-- #region -->
# PARTIE 2: Evolution des débits de l'Arve


Dans cet [article](https://doi.org/10.1038/s41598-020-67379-7), les autrices et les auteurs examinent l'impact du changement climatique et de la perte de masse des glaciers sur l'hydrologie du massif du Mont-Blanc, en particulier sur le bassin versant de la rivière Arve, qui est alimentée par une série de glaciers dans la région. Ils ont utilisé des projections climatiques (scénarios RCP4.5 et RCP8.5) et des simulations de dynamique des glaciers (historiques et futures) combinées a un modèle hydrologique pour étudier l'évolution du débit des rivières à l'échelle du 21e siècle (vus en cours).

Vous avez représenté ci-dessous la zone d'étude de l'article avec en panneau (a), la localisation du bassin versant de Sallanches dans les Alpes françaises et la carte des bassins étudiés et en panneau (b), le massif du Mont-Blanc vu de Téléphérique de la Flégère (point noir avec angle de vue sur le plan).
![Zone étudiée: Fig 1 du papier](Fig1.png)

<!-- #endregion -->

<!-- #region -->
Dans leur article, ils commencent par étudier l'évolution des températures et des précipitations à Sallances. Ils montrent qu'il y a une nette augmentation des températures estivales et hivernales dans la région du Mont-Blanc et des modificaions de précipitations. Cela correspond à la figure 2 représentée ci-dessous. **Avez-vous obtenus les mêmes résultats avec votre analyse de la partie 1?**


![Fig 2 du papier](Fig2.png)


<!-- #endregion -->

Dans cette partie, nous allons reproduire la figure n°3 du papier qui illustre l'évolution des débits saisonniers de la rivière Arve en fonction des scénarios climatiques. La figure présente des courbes pour les débits moyens (en hiver et en été) simulées sous les scénarios RCP4.5 et RCP8.5, à partir de différents modèles climatiques pour la région du Mont-Blanc.


## Chargement des données

Dans le dossier `Donnees_Debits/`, vous disposez de données simulées historiques ainsi que de projections futures des débits cumulés (et de leur écart-type) de la rivière Arve (pour chaque scénario climatique). Ces données sont organisées en fonction des saisons (hiver, été ou moyenne annuelle) et des différents scénarios climatiques (historiques, RCP4.5, RCP8.5). 
Vous disposez également des séries temporelles de débits observés (`Debits_obs_Sal_sans_2902`) pour la saison d'hiver et d'été, qui servent de référence pour la comparaison des simulations.


**Regardez un peu les données. Quelles sont les dimensions des données historiques et des données de modélisation ?**

*Indications: Pour rappel, les deux scénarios RCP font partie des trajectoires d'émissions de gaz à effet de serre utilisées pour projeter les futurs changements climatiques. Pour chacun d'eux, plusieurs modèles climatiques régionaux ont été utilisés pour simuler l'évolution du climat et ensuite du débit de la rivière. Les différents modèles régionaux sont des modèles climatiques spécifiques, souvent *downscalés* (affinés à une échelle régionale) pour mieux simuler les conditions climatiques locales du Mont Blanc. Le processus de downscaling permet d'obtenir des projections climatiques à une résolution spatiale plus fine que celle des modèles climatiques globaux.* 


```python
import pandas as pd
import matplotlib.pyplot as plt
import os

# Dossier contenant les fichiers
data_folder = "Donnees_Debits/"

# Fonction 1 : Chargement des fichiers selon le scénario et la saison
def load_files(scenario, season):
    """
    Charge les fichiers correspondant au scénario et à la saison donnés.
    :param scenario: Scénario climatique ("historic", "rcp45", "rcp85").
    :param season: Saison ("hiver", "ete", "annuele").
    :return: Deux DataFrames (cumulé et SD).
    """
    # Construire les noms de fichiers
    cum_file = f"Donnees_Debits/Q_cum_{scenario}_{season}_multimodeles_OK"
    sd_file = f"Donnees_Debits/Q_sd_{scenario}_{season}_multimodeles_OK"

    # Charger les fichiers

    

    cum_data = pd.read_csv(cum_file, sep=',', parse_dates=['Date'])
    sd_data = pd.read_csv(sd_file, sep=',', parse_dates=['Date'])

    return cum_data, sd_data
```

## Analyse et création de fonctions

**Etant donné une saison, faites une fonction qui pour chaque scenario (historique, ou RCP), calcule la moyenne et l'écart-type moyen du débit pour chaque année. Commencez par charger les données et transformer la date pour pouvoir extraire l'année. Dans un second temps, chargez les données observées et extrayez la valeur pour la saison considérée. Puis, reproduisez les graphiques montrant l'évolution du débit moyen pour les saisons d'hiver et d'été, sous les deux scénarios RCP4.5 et RCP8.5, comparés aux données historiques et aux données observées.**

Est-ce clair pour vous, quelle est la différence entre les données observées et historiques ?



**Discutez comment le changement climatique impacte les ressources en eau dans les régions montagneuses.**



```python
def visualize_discharge(scenario, season):
    """
    Visualise les débits moyens et SD pour un scénario et une saison donnés.
    :param scenario: Scénario climatique ("historic", "rcp45", "rcp85").
    :param season: Saison ("hiver", "ete", "annuele").
    """
    # Charger les fichiers
    cum_data, sd_data = load_files(scenario, season)

  

    # Calcul des moyennes annuelles
    cum_data_annual = cum_data.groupby('Date').mean().reset_index()
    sd_data_annual = sd_data.groupby('Date').mean().reset_index()

    # Calculer la moyenne des colonnes de débit
    mean_discharge = cum_data_annual.iloc[:, 1:].mean(axis=1)
    sd_discharge = sd_data_annual.iloc[:, 1:].mean(axis=1)

    # Visualisation avec intervalle SD
    plt.figure(figsize=(12, 6))
    plt.plot(cum_data_annual['Date'], mean_discharge, label=f'{scenario.upper()} - Mean', linestyle='-', marker='o', color='blue')
    plt.fill_between(cum_data_annual['Date'], 
                     mean_discharge - 2 * sd_discharge, 
                     mean_discharge + 2 * sd_discharge, 
                     color='blue', alpha=0.2, label=f'{scenario.upper()} - SD Interval')
    plt.title(f'Évolution des débits moyens et SD ({season.capitalize()}) - {scenario.upper()}')
    plt.xlabel('Année')
    plt.ylabel('Débit (m³/s)')
    plt.legend()
    plt.grid()
    plt.show()

# Visualisation groupée par saison
def visualize_by_season(season):
    """
    Visualise les débits pour tous les scénarios dans une saison donnée.
    :param season: Saison ("hiver", "ete", "annuele").
    """
    plt.figure(figsize=(14, 8))
    for scenario in ["hist", "RCP4-5", "RCP8-5"]:
        cum_data, sd_data = load_files(scenario, season)

        
        # Calcul des moyennes annuelles
        cum_data_annual = cum_data.groupby('Date').mean().reset_index()
        sd_data_annual = sd_data.groupby('Date').mean().reset_index()

        # Calculer la moyenne des colonnes de débit
        mean_discharge = cum_data_annual.iloc[:, 1:].mean(axis=1)
        sd_discharge = sd_data_annual.iloc[:, 1:].mean(axis=1)

        # Ajouter les courbes avec intervalle SD
        plt.plot(cum_data_annual['Date'], mean_discharge, label=f'{scenario.upper()} - Mean ({season})', linestyle='-', marker='o')
        plt.fill_between(cum_data_annual['Date'], 
                         mean_discharge - 15 * sd_discharge, 
                         mean_discharge + 15 * sd_discharge, 
                         alpha=0.2, label=f'{scenario.upper()} - SD Interval ({season})')

    plt.title(f'Évolution des débits moyens et SD - {season.capitalize()}')
    plt.xlabel('Année')
    plt.ylabel('Débit (m³/s)')
    plt.legend()
    plt.grid()
    plt.show()

# Exemple d'utilisation
visualize_by_season("hiver")
visualize_by_season("ete")
visualize_by_season("annee")
```

```python

```

```python

```
