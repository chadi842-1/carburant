{{ config(
    partition_by={'field': 'date_prix', 'data_type': 'date'} if target.type == 'bigquery' else none,
    cluster_by=['carburant_id', 'pdv_id'] if target.type == 'bigquery' else none
) }}
-- Fact table: one valid price observation per station, fuel and update timestamp.
select
    staging_row_id                 as prix_id,
    pdv_id,
    carburant_id,
    cast(maj as date)              as date_prix,
    maj                            as horodatage,
    prix_eur
from {{ ref('stg_prix') }}
where reason is null
