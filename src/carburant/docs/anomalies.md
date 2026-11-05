# Détection d'anomalies de prix

## Définition d'un prix suspect
Un prix est suspect lorsqu'il s'écarte à la fois de l'historique récent de la station et des prix de ses pairs le même jour, c'est-à-dire les stations du même département, pour le même carburant et le même type de route. Une hausse générale du marché déplace toutes les stations ensemble et n'est donc pas signalée.

## Méthodes comparées
Deux détecteurs utilisent les mêmes variables. La règle robuste calcule des z-scores fondés sur la médiane et l'écart absolu médian, puis retient le plus petit des deux écarts, historique et pairs. L'Isolation Forest est un modèle non supervisé entraîné sur la première moitié de la période.

## Protocole d'évaluation
Comme il n'existe pas d'étiquettes réelles, des anomalies de 3 à 20 % sont injectées dans 1 % des prix journaliers. Les jours sont découpés dans le temps : 50 % pour l'ajustement, 25 % pour choisir le seuil qui maximise le F1, et 25 % pour la mesure finale de la précision et du rappel.

## Limites
Les petites anomalies, de 3 à 5 %, sont difficiles à distinguer de la variation normale d'une station et sont rarement détectées. Les anomalies injectées sont ponctuelles : une manipulation durable et progressive ne serait pas détectée de la même façon.
