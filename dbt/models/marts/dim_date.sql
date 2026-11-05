-- One row per calendar day between the first and the last price observation.
with bounds as (
    select min(date_prix) as d0, max(date_prix) as d1 from {{ ref('fct_prix') }}
),

days as (
    select cast(unnest(generate_series(d0, d1, interval 1 day)) as date) as date_prix
    from bounds
)

select
    date_prix,
    extract(year from date_prix)        as annee,
    extract(month from date_prix)       as mois,
    extract(week from date_prix)        as semaine_iso,
    extract(isodow from date_prix)      as jour_semaine,
    extract(isodow from date_prix) >= 6 as week_end
from days
