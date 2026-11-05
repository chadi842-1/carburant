-- Aggregate built for a BI tool (Looker Studio / Power BI): daily price by département and fuel.
select
    date_prix,
    region,
    departement,
    carburant,
    count(*)                       as n_stations,
    round(avg(prix_eur), 4)        as prix_moyen,
    round(min(prix_eur), 3)        as prix_min,
    round(max(prix_eur), 3)        as prix_max
from {{ ref('mart_prix_journalier') }}
group by all
