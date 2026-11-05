-- Every staged price row ends up in exactly one of fct_prix or quarantine_prix.
with counts as (
    select
        (select count(*) from {{ source('raw', 'prix') }})        as staged,
        (select count(*) from {{ ref('fct_prix') }})              as fact,
        (select count(*) from {{ ref('quarantine_prix') }})       as quarantined
)
select * from counts where staged <> fact + quarantined
