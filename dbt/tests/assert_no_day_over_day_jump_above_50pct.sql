-- Severity warn: a valid price that moves more than 50% from the station's previous price is
-- suspicious but not impossible, so it is reported, not blocked. (Subtler manipulation is the
-- anomaly detector's job.)
{{ config(severity='warn') }}
with ordered as (
    select
        pdv_id, carburant_id, horodatage, prix_eur,
        lag(prix_eur) over (partition by pdv_id, carburant_id order by horodatage) as prev
    from {{ ref('fct_prix') }}
)
select * from ordered
where prev is not null and abs(prix_eur / prev - 1) > 0.5
