# Architecture de la plateforme

## Vue d'ensemble du pipeline
Le pipeline suit une approche ELT en cinq étapes : ingestion des fichiers XML publiés, mise en zone brute (raw) partitionnée par date, parsing vers des fichiers parquet de staging, chargement sans transformation dans le schéma raw de l'entrepôt, puis transformations et tests de qualité avec dbt. Chaque étape est idempotente : relancer une journée remplace ses fichiers sans créer de doublons.

## Zone brute et stockage
Les fichiers sont rangés comme dans un bucket Cloud Storage : lake/raw/date=AAAA-MM-JJ/prix.xml. La zone brute conserve les fichiers tels qu'ils ont été publiés, ce qui permet de rejouer tout le pipeline depuis l'origine si une règle de transformation change.

## Entrepôt et modèle en étoile
L'entrepôt est DuckDB en local et BigQuery en production, avec le même projet dbt. La table de faits fct_prix contient une ligne par station, carburant et horodatage de mise à jour. Les dimensions sont dim_station, dim_carburant, dim_date et dim_localisation. En BigQuery, fct_prix est partitionnée par date_prix et clusterisée par carburant_id et pdv_id, ce qui réduit le volume scanné par les requêtes filtrées sur une période.

## Orchestration
Un DAG Airflow quotidien enchaîne l'ingestion, le staging, le chargement, dbt build puis la détection d'anomalies. Si un test dbt bloquant échoue, les tâches suivantes ne s'exécutent pas et les données de la veille restent exposées.

## Intégration continue
GitHub Actions lance le linting (ruff) et la suite de tests (pytest, qui construit un petit entrepôt et exécute dbt build) à chaque push et à chaque pull request.
