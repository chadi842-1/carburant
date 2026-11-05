-- Last valid price of the day per station and fuel, joined to its dimensions.
-- Feeds the dashboard, the anomaly detector and the agent's SQL tool.
with ranked as (
    select
        f.*,
        row_number() over (
            partition by pdv_id, carburant_id, date_prix order by horodatage desc
        ) as rn
    from {{ ref('fct_prix') }} f
)

select
    r.date_prix,
    r.pdv_id,
    s.ville,
    s.cp,
    s.dep,
    l.departement,
    l.region,
    s.autoroute,
    c.carburant,
    r.prix_eur
from ranked r
join {{ ref('dim_station') }} s using (pdv_id)
left join {{ ref('dim_localisation') }} l using (dep)
join {{ ref('dim_carburant') }} c using (carburant_id)
where r.rn = 1
