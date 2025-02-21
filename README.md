# Labo 0 : Reproduction des résultats de Paul Baran

Ce projet vise à reproduire les résultats fondamentaux de la simulation implémentée par Paul Baran dans son article "On Distributed Communication Networks". 

L'objectif est de démontrer la robustesse des réseaux distribués en simulant des pannes de nœuds et en observant le niveau de survie du réseau.

## Objectifs

- **Reproductibilité des résultats** : Simuler les résultats de Paul Baran en utilisant la méthode Monte-Carlo.
- **Analyse de la robustesse** : Étudier l'impact de la redondance et de la probabilité de panne sur la survie du réseau.

## Méthodologie

### 1. Probabilité de panne et niveau de redondance

- **Redondance (R)** : Ratio du nombre de liens dans la topologie du graphe sur le nombre minimal de liens nécessaires pour relier tous les nœuds.
- **Probabilité de panne (P_p)** : Probabilité qu'un nœud tombe en panne, variant de 0 à 1.
- **Topologies** : Ligne (R=1), Grille (R=2), Grille+ (R=3), Grille++ (R=4).

### 2. Niveau de survie du réseau

**Le niveau de survie (S)** est calculé comme le pourcentage de nœuds présents dans la plus grande composante connexe du graphe après la simulation de panne.


NetworkX