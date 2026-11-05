# Qualité des données

## Règles de quarantaine
Chaque ligne de prix est vérifiée par le modèle dbt stg_prix. La première règle enfreinte donne la raison de quarantaine : unknown_fuel si le carburant n'appartient pas au catalogue officiel, bad_timestamp si la date de mise à jour est illisible, missing_price si la valeur est vide, out_of_range si le prix est hors de la plage plausible de 0,30 à 4,00 euros par litre, duplicate si la même publication apparaît plusieurs fois. Les lignes valides vont dans fct_prix, les autres dans quarantine_prix avec leur raison.

## Pourquoi mettre en quarantaine plutôt que supprimer
Une ligne en quarantaine reste consultable et comptée. On peut mesurer la part de lignes invalides, expliquer chaque exclusion et corriger une règle trop stricte sans recharger les données. Un test dbt vérifie que chaque ligne de staging se retrouve exactement une fois, soit dans la table de faits, soit en quarantaine.

## Tests dbt
Le projet déclare des tests d'unicité, de non-nullité, d'intégrité référentielle entre faits et dimensions, de valeurs acceptées, de plage de prix et d'unicité de la combinaison station, carburant, horodatage. Un test en sévérité warn signale les variations de plus de 50 % d'un jour à l'autre sans bloquer le pipeline.

## Erreur de saisie typique
Une erreur fréquente est le glissement de virgule : 17,99 au lieu de 1,799. Le prix sort alors de la plage plausible et la ligne part en quarantaine avec la raison out_of_range.
