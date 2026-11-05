-- Rows kept out of the fact table, with the rule they broke. Never exposed to the dashboard.
select
    staging_row_id,
    pdv_id,
    carburant,
    maj_raw,
    valeur_raw,
    prix_eur,
    source_file,
    reason
from {{ ref('stg_prix') }}
where reason is not null
