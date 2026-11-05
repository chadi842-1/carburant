-- Every staged price row, typed, with the FIRST quality rule it breaks (null = valid).
-- Rule order matters: a row is quarantined for one reason only.
with src as (
    select
        pdv_id,
        carburant_id,
        carburant,
        maj_raw,
        cast(maj as timestamp)       as maj,
        valeur_raw,
        cast(prix_eur as double)     as prix_eur,
        source_file
    from {{ source('raw', 'prix') }}
),

flagged as (
    select
        *,
        row_number() over (
            partition by pdv_id, carburant, maj_raw, coalesce(valeur_raw, '')
            order by source_file
        ) as copy_number
    from src
)

select
    {{ dbt_utils_like_hash(['pdv_id', 'carburant', 'maj_raw', 'valeur_raw', 'source_file',
                            'copy_number']) }} as staging_row_id,
    pdv_id,
    carburant_id,
    carburant,
    maj_raw,
    maj,
    valeur_raw,
    prix_eur,
    source_file,
    case
        when carburant_id is null
             or carburant not in ('Gazole', 'SP95', 'SP98', 'E10', 'E85', 'GPLc')
            then 'unknown_fuel'
        when maj is null then 'bad_timestamp'
        when prix_eur is null then 'missing_price'
        when prix_eur < {{ var('price_min') }} or prix_eur > {{ var('price_max') }}
            then 'out_of_range'
        when copy_number > 1 then 'duplicate'
    end as reason
from flagged
