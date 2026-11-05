"""Learned intent classifier for the agent (router v2), plus improved slot extraction.

Training data is GENERATED from templates and slot fillers (no question from either evaluation
set is used). The model is TF-IDF on word 1-2-grams and character 2-5-grams, then a multinomial
logistic regression. A question whose top probability is below `threshold` is declined.

Caveat stated plainly: the same person wrote the templates and the evaluation questions, so the
measured accuracy is optimistic compared with real users' phrasing. To limit leakage, the
out-of-scope training topics avoid the subjects used in the test set.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import SEED
from .rag import normalize

INTENTS = ["cheapest", "most_expensive", "average", "trend", "count_stations", "motorway_gap",
           "anomalies", "doc", "out_of_scope"]

FUELS = ["gazole", "diesel", "gasoil", "SP95", "sans plomb 95", "SP98", "sans plomb 98", "E10",
         "E85", "superéthanol", "GPL", "GPLc"]
PLACES = ["Paris", "Lyon", "Marseille", "Toulouse", "Nice", "Nantes", "Bordeaux", "Lille",
          "Rennes", "Strasbourg", "Brest", "Limoges", "Orléans", "Nanterre", "Clermont-Ferrand",
          "Bretagne", "Occitanie", "Grand Est", "Normandie", "Île-de-France", "le Nord",
          "la Gironde", "le Rhône", "le département 13", "le département 75", "France",
          "Pays de la Loire", "Hauts-de-France", "l'Isère", "le Loiret"]
PERIODS = ["", "cette semaine", "aujourd'hui", "ce mois", "sur les 30 derniers jours",
           "depuis 15 jours", "en ce moment", "ces 7 derniers jours"]

TEMPLATES: dict[str, list[str]] = {
    "cheapest": [
        "station la moins chère pour le {f} à {p} {t}", "où trouver du {f} pas cher à {p}",
        "le {f} le moins cher à {p} {t}", "meilleur prix du {f} à {p}",
        "les {k} stations les moins chères en {f} à {p}", "quelle station a le prix le plus bas "
        "en {f} à {p}", "où faire le plein de {f} au meilleur prix à {p}", "top {k} des "
        "stations les moins chères pour le {f} à {p}", "{f} pas cher {p}",
        "station la moins chère sur autoroute pour le {f}", "je cherche du {f} bon marché à {p}",
    ],
    "most_expensive": [
        "station la plus chère pour le {f} à {p} {t}", "où le {f} est le plus cher à {p}",
        "les {k} stations les plus chères en {f} à {p}", "prix le plus élevé du {f} à {p}",
        "quelle station vend le {f} le plus cher à {p}", "top {k} des stations les plus chères "
        "en {f} à {p}", "tarif maximum du {f} à {p} {t}", "{f} le plus cher {p}",
    ],
    "average": [
        "prix moyen du {f} à {p} {t}", "combien coûte le {f} à {p}", "quel est le prix du {f} "
        "à {p} {t}", "à combien est le {f} à {p}", "le prix du {f} à {p}",
        "combien vaut le litre de {f} à {p}", "prix moyen du {f} sur autoroute",
        "en moyenne combien coûte le {f} en {p}", "tarif moyen du {f} {t}",
        "combien je paie le litre de {f} à {p}",
    ],
    "trend": [
        "évolution du prix du {f} à {p} {t}", "le {f} a-t-il augmenté à {p} {t}",
        "le {f} a-t-il baissé à {p}", "tendance du {f} en {p} {t}", "le prix du {f} monte ou "
        "descend à {p}", "est-ce que le {f} est en hausse à {p}", "comment a évolué le {f} "
        "à {p} {t}", "variation du prix du {f} {t}", "le {f} baisse-t-il en {p}",
    ],
    "count_stations": [
        "combien de stations à {p}", "nombre de stations à {p}", "combien de stations vendent "
        "du {f} à {p}", "nombre de points de vente de {f} en {p}", "combien de stations-service "
        "proposent du {f} à {p}", "combien y a-t-il de stations essence à {p}",
    ],
    "motorway_gap": [
        "écart de prix entre autoroute et route pour le {f}", "le {f} est-il plus cher sur "
        "autoroute à {p}", "surcoût de l'autoroute pour le {f}", "différence de prix autoroute "
        "route {f} {p}", "combien plus cher sur autoroute pour le {f}", "faire le plein sur "
        "autoroute coûte combien de plus en {f}", "comparaison autoroute et route pour le {f}",
    ],
    "anomalies": [
        "y a-t-il des prix suspects à {p} {t}", "anomalies de prix à {p} {t}", "combien de prix "
        "anormaux pour le {f} {t}", "liste des anomalies de prix en {p}", "prix aberrants "
        "détectés à {p}", "des prix bizarres à {p} {t}", "prix suspects pour le {f} en {p}",
        "stations avec des prix anormaux à {p}",
    ],
}

DOC_QUESTIONS = [
    "comment fonctionne le pipeline", "pourquoi mettre en quarantaine",
    "qu'est-ce que la quarantaine", "quelles sont les règles de qualité",
    "comment sont testées les données avec dbt", "que signifie l'attribut pop",
    "quel est le format des fichiers xml", "en quelle unité sont les prix",
    "quelle est l'unité de la latitude", "comment est orchestré le pipeline", "que fait airflow",
    "comment fonctionne l'intégration continue", "dans quel dossier vont les fichiers téléchargés",
    "qu'est-ce que la zone brute", "comment est conçu l'entrepôt",
    "qu'est-ce que le modèle en étoile", "pourquoi partitionner la table de faits",
    "quelle est la table de faits", "comment est défini un prix suspect",
    "quelles méthodes de détection sont comparées", "comment est évaluée la détection",
    "quelles sont les limites du détecteur", "pourquoi les petites anomalies sont difficiles",
    "comment fonctionne l'agent", "à quoi sert l'outil sql", "pourquoi pas de sql libre",
    "comment marche la recherche documentaire", "c'est quoi bm25",
    "les données sont-elles réelles", "d'où viennent les données", "quels carburants existent",
    "qu'est-ce que le e85", "qu'est-ce que le gplc", "que contient le e10",
    "comment est évalué l'assistant", "quelles erreurs de saisie sont détectées",
    "que se passe-t-il si un test échoue", "quels tests dbt existent",
    "comment sont gérés les doublons", "comment sont traités les prix hors plage",
    "pourquoi comparer aux stations voisines", "quel encodage pour les fichiers",
    "comment rejouer le pipeline", "qu'est-ce qu'une étape idempotente",
]

OUT_OF_SCOPE = [
    "quel temps fait-il demain", "météo à {p} ce week-end", "qui a gagné le match hier",
    "résultat du tour de france", "recommande-moi un film", "quels films au cinéma cette "
    "semaine", "comment soigner un rhume", "quels sont les symptômes de la grippe", "écris une "
    "fonction python qui trie une liste", "comment installer docker", "traduis bonjour en anglais",
    "combien coûte un billet de train pour {p}", "prix d'un billet d'avion pour {p}", "combien "
    "coûte une voiture d'occasion", "prix d'un vélo électrique", "combien coûte un téléphone",
    "combien coûte un café à {p}", "quel est le prix de l'électricité", "prix du gaz pour le "
    "chauffage", "quel est le cours de l'action total", "combien vaut le dollar en euros", "quand "
    "a été construite la tour eiffel", "qui était napoléon", "calcule 17 fois 23", "raconte une "
    "blague", "quel âge as-tu", "combien de musées à {p}", "combien d'habitants à {p}", "combien "
    "d'écoles à {p}", "hôtel pas cher à {p}", "restaurant le moins cher à {p}", "loyer moyen "
    "d'un studio", "salaire moyen d'un ingénieur", "quelle est la capitale de l'australie",
    "comment faire une pâte à crêpes", "horaires de la piscine à {p}", "nombre de pharmacies "
    "de garde à {p}", "prix moyen d'une maison en {p}", "taux d'intérêt du crédit immobilier",
]


def _fill(tpl: str, rng: np.random.Generator) -> str:
    return re.sub(r"\s+", " ", tpl.format(
        f=FUELS[rng.integers(len(FUELS))], p=PLACES[rng.integers(len(PLACES))],
        t=PERIODS[rng.integers(len(PERIODS))], k=["2", "3", "4", "5", "trois", "cinq"][
            rng.integers(6)])).strip()


def training_data(per_template: int = 12, seed: int = SEED) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    X, y = [], []
    for intent, tpls in TEMPLATES.items():
        for tpl in tpls:
            for _ in range(per_template):
                X.append(_fill(tpl, rng))
                y.append(intent)
    for q, stem in itertools.product(DOC_QUESTIONS, ["", " ?", "explique-moi "]):
        X.append(f"{stem}{q}" if stem.startswith("explique") else f"{q}{stem}")
        y.append("doc")
    for tpl in OUT_OF_SCOPE:
        for _ in range(3):
            X.append(_fill(tpl, rng))
            y.append("out_of_scope")
    return X, y


def build_model(C: float = 4.0) -> Pipeline:
    feats = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                                 sublinear_tf=True, min_df=2)),
    ])
    return Pipeline([("tfidf", feats),
                     ("clf", LogisticRegression(C=C, max_iter=3000, random_state=SEED))])


@dataclass
class IntentModel:
    pipeline: Pipeline
    threshold: float
    cv_accuracy: float

    def predict(self, question: str) -> tuple[str, float]:
        p = self.pipeline.predict_proba([question])[0]
        i = int(np.argmax(p))
        return self.pipeline.classes_[i], float(p[i])


def train(threshold: float = 0.5, C: float = 4.0, seed: int = SEED) -> IntentModel:
    """threshold 0.5: middle of the 0.45-0.55 plateau of routing accuracy on the DEV set."""
    X, y = training_data(seed=seed)
    cv = cross_val_score(build_model(C), X, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
    return IntentModel(build_model(C).fit(X, y), threshold, float(cv))


# ----------------------------------------------------------------- improved slot extraction
FUEL_PATTERNS_V2: list[tuple[str, str]] = [
    (r"\bsp\s?95[\s-]?e\s?10\b", "E10"),                    # official name of E10
    (r"\bsp\s?98\b|sans plomb 98|super 98", "SP98"),
    (r"\bsp\s?95\b|sans plomb 95|sans plomb\b", "SP95"),
    (r"\be\s?10\b", "E10"),
    (r"\be\s?85\b|superethanol|bioethanol", "E85"),
    (r"\bgplc?\b|\bgaz de petrole", "GPLc"),
    (r"\bgazole\b|\bdiesel\b|\bgasoil\b", "Gazole"),
]
NUMBER_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
                "dix": 10}


def parse_fuel_v2(q: str) -> str | None:
    for pat, fuel in FUEL_PATTERNS_V2:
        if re.search(pat, q):
            return fuel
    return None


def parse_k_v2(q: str) -> int:
    num = r"(\d+|un|une|deux|trois|quatre|cinq|six|dix)"
    for pat in (rf"\btop\s*{num}\b", rf"\b{num} (?:premieres |premiers |meilleures )?"
                rf"(?:stations|points de vente)"):
        m = re.search(pat, q)
        if m:
            v = m.group(1)
            return max(1, min(int(v) if v.isdigit() else NUMBER_WORDS[v], 10))
    return 1


def parse_days_v2(q: str) -> int:
    if "aujourd" in q:
        return 1
    m = re.search(r"(\d+) (?:derniers )?jours", q)
    if m:
        return max(1, int(m.group(1)) if int(m.group(1)) != 15 else 14)
    if re.search(r"\bce mois\b|dernier mois|un mois|\b30 jours", q):
        return 30
    if re.search(r"deux semaines|quinzaine", q):
        return 14
    return 7


def parse_autoroute(q: str, intent: str) -> bool:
    """Restrict to motorway stations when asked, except for the motorway-vs-road comparison."""
    return intent != "motorway_gap" and bool(re.search(r"\bautoroute", q))
