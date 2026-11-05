select
    pdv_id,
    adresse,
    ville,
    cp,
    dep,
    autoroute,
    latitude,
    longitude
from {{ ref('stg_stations') }}
