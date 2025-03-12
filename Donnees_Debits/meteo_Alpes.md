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
```

**Gestion des valeurs manquantes et filtration**

- Quelles colonnes allez vous sélectionner pour notre étude ? Pour rappel, on va étudier l'évolution de la température et des précipitations depuis les années 1950 dans les Alpes.
*Indication:* La localisation des postes pourra être utile.
- Créer la table `meteo_cleaned` avec les colonnes d'interet et sans données manquantes.
- Faites une première analyse sur les informations statistiques de base sur les données (moyenne, médiane, écart-type, etc.).


```python
#Votre code ici
nombre_de_postes = meteo['NUM_POSTE'].nunique()
print(f"Le nombre de postes distincts est : {nombre_de_postes}")
```

- Combien de stations restent dans notre étude ? Où se situent-elles ?

```python
#Votmeteo = pd.read_csv("grt.csv")  re code ici 
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
#Votre code ici
```

### Prévisions

Quelle température fera-t-il en 2100 selon votre modèle ?


```python
#Votre code ici
```

### Variabilité saisonnière 

Analysez la variabilité saisonnière des données. 

- Manipulez les données pour les regrouper mois par mois. Y a-t-il des tendances ? Correspondent-elles à vos connaissances ?
- N'oubliez pas de visualisez les données
- Enfin si vous êtes à l'aise, étudiez en plus les épisodes de fortes chaleurs (températures > 28°C). Est-ce qu'il y en a plus souvent plus récemment?



```python
#Votre code ici

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
#Votre code ici
```

## Analyse et création de fonctions

**Etant donné une saison, faites une fonction qui pour chaque scenario (historique, ou RCP), calcule la moyenne et l'écart-type moyen du débit pour chaque année. Commencez par charger les données et transformer la date pour pouvoir extraire l'année. Dans un second temps, chargez les données observées et extrayez la valeur pour la saison considérée. Puis, reproduisez les graphiques montrant l'évolution du débit moyen pour les saisons d'hiver et d'été, sous les deux scénarios RCP4.5 et RCP8.5, comparés aux données historiques et aux données observées.**

Est-ce clair pour vous, quelle est la différence entre les données observées et historiques ?



**Discutez comment le changement climatique impacte les ressources en eau dans les régions montagneuses.**



```python
#Votre code ici
```
