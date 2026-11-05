# Données sources

## Format des fichiers officiels
Les prix proviennent du jeu de données ouvert « Prix des carburants en France » publié sur data.gouv.fr, au format XML encodé en ISO-8859-1. Chaque point de vente est un élément pdv avec les attributs id, latitude, longitude, cp et pop, et contient des éléments prix avec les attributs nom, id, maj et valeur.

## Unités
La latitude et la longitude sont publiées multipliées par 100000. Le prix est documenté en euros multipliés par 1000 (1799 pour 1,799 euro), mais des fichiers récents utilisent directement des euros décimaux ; le parser accepte les deux formes.

## Carburants
Le catalogue officiel compte six carburants : Gazole, SP95, SP98, E10, E85 et GPLc. Le E10 contient jusqu'à 10 % d'éthanol, le E85 est le superéthanol et le GPLc est le gaz de pétrole liquéfié carburant.

## Station sur autoroute
L'attribut pop vaut A pour une station sur autoroute et R pour une station sur route. Les stations d'autoroute pratiquent en général des prix plus élevés, c'est pourquoi la détection d'anomalies compare une station à des pairs du même type de route.

## Données synthétiques
L'environnement de développement n'a pas accès au flux officiel. Le pipeline est donc exécuté sur des fichiers synthétiques qui respectent exactement le format officiel, couvrent 20 départements et contiennent des défauts injectés volontairement. Les chiffres publiés décrivent cette simulation et non le marché réel.
