# Reproduction des Résultats de Paul Baran

## Contexte

Ce projet est un **laboratoire académique** réalisé dans le cadre de mon cours d'architecture des réseaux à l'HEPIA.  

Il vise à **reproduire les résultats fondateurs de Paul Baran** sur la robustesse des réseaux distribués présentés dans son article *On Distributed Communication Networks*.

Paul Baran a démontré dès les années 1960 que des réseaux construits avec des composants fragiles pouvaient néanmoins être extrêmement robustes grâce à la **redondance des liens**.  

Le but de ce projet est de reproduire expérimentalement une partie de ses résultats (figure 4 de l’article) à l’aide de simulations Monte-Carlo, en étudiant différentes topologies de réseaux et leur résistance face aux pannes de nœuds et de liens.

## Contenu du dépôt

- `Rapport_Baran.pdf` : mon rapport détaillant la méthodologie, les résultats et l’analyse critique.
- `main.py` : script principal pour lancer la simulation.
- `simulation.py` : gestion des défaillances, calcul du niveau de survie et mesures statistiques.
- `topologies.py` : génération des différentes topologies de réseaux (ligne, grille, grille+, grille++, anneau, étoile, arbre, hybride).
- `visualisation.py` : visualisation des topologies, de l’importance des nœuds et des résultats de simulation avec `matplotlib`.

## Exécution

### Prérequis

- Python **3.8+**
- Dépendances Python :
  ```bash
  pip install -r requirements.txt
  ```

### Lancer une simulation

Pour exécuter la simulation avec les paramètres par défaut :

```bash
python3 main.py
```

Vous pouvez également personnaliser l’exécution en ajoutant des arguments :

```bash
python3 main.py -s 18 -p 25 -n 15 -v 6
```

-s : taille de la grille (par défaut 18, soit un réseau de 18×18 nœuds comme dans l’article de Baran)
-p : nombre de points de probabilité de panne à tester (par défaut 25)
-n : nombre d’essais Monte-Carlo par probabilité (par défaut 15)
-v : taille réduite utilisée pour la visualisation des topologies (par défaut 6)

### Exemple de sortie

Le programme génère :

- Les visualisations des topologies avec importance des nœuds (centralité en degré + intermédiaire).

- Les courbes comparant le niveau de survie du réseau en fonction de la probabilité de panne pour chaque topologie.

## Résultats

Les résultats complets, analyses et comparaisons avec ceux de Paul Baran sont disponibles dans le fichier [Rapport_Baran.pdf](Rapport_Baran.pdf).