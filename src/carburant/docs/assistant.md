# Assistant et agent

## Fonctionnement de l'agent
L'agent est un graphe LangGraph à trois étapes : un routeur qui classe la question, un outil qui y répond, puis une étape de rédaction qui cite ses sources. Les questions chiffrées passent par l'outil SQL, les questions sur la méthode passent par la recherche documentaire, et les questions hors sujet reçoivent une réponse d'abstention.

## Outil SQL
L'outil SQL n'exécute que des requêtes paramétrées prédéfinies, en lecture seule, sur les tables mart_prix_journalier et anomalies_prix. Il ne génère jamais de SQL libre, ce qui écarte les injections SQL et les requêtes coûteuses. La période « cette semaine » désigne les sept derniers jours présents dans l'entrepôt.

## Recherche documentaire
La recherche documentaire utilise l'algorithme BM25 sur des extraits de cette documentation découpée par section. Si aucun extrait n'est assez pertinent, l'agent préfère répondre qu'il ne sait pas plutôt que d'inventer.

## Absence de modèle de langage
Le routeur et la rédaction sont déterministes : aucun modèle de langage n'est appelé, ce qui rend les réponses reproductibles et testables. Un modèle de langage peut être branché pour reformuler la réponse à partir des résultats déjà calculés, sans pouvoir modifier les chiffres.

## Évaluation
L'agent est évalué sur un jeu de questions écrit à la main, avec des réponses attendues calculées indépendamment de l'outil SQL. On mesure la justesse, la part de réponses appuyées sur une source et la latence.
