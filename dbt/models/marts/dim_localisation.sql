-- Département -> name and region. Départements seen in the data but missing from the reference
-- seed are kept with an 'inconnu' label so the station dimension never loses a foreign key.
with seen as (
    select distinct dep from {{ ref('stg_stations') }} where dep is not null
)

select
    s.dep,
    coalesce(d.nom, 'inconnu')     as departement,
    coalesce(d.region, 'inconnu')  as region
from seen s
left join {{ ref('departements') }} d on d.dep = s.dep
