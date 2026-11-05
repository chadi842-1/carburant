-- Latest description of each station across all published files.
with ranked as (
    select
        *,
        row_number() over (partition by pdv_id order by source_file desc) as rn
    from {{ source('raw', 'stations') }}
)

select
    pdv_id,
    adresse,
    ville,
    cp,
    case
        when cp is null or length(cp) <> 5 then null
        when substr(cp, 1, 2) = '20' then case when cast(cp as integer) < 20200 then '2A' else '2B' end
        when substr(cp, 1, 2) = '97' then substr(cp, 1, 3)
        else substr(cp, 1, 2)
    end                                   as dep,
    pop = 'A'                             as autoroute,
    case when latitude between -90 and 90 and latitude <> 0 then latitude end   as latitude,
    case when longitude between -180 and 180 and longitude <> 0 then longitude end as longitude
from ranked
where rn = 1
