{# Portable surrogate key: md5 of the concatenated columns (works on DuckDB and BigQuery). #}
{% macro dbt_utils_like_hash(columns) -%}
    {%- if target.type == 'bigquery' -%}
    to_hex(md5(concat(
    {%- else -%}
    md5(concat(
    {%- endif -%}
    {%- for c in columns -%}
        coalesce(cast({{ c }} as {{ 'string' if target.type == 'bigquery' else 'varchar' }}), '∅')
        {%- if not loop.last %}, '|', {% endif -%}
    {%- endfor -%}
    {%- if target.type == 'bigquery' -%} ))) {%- else -%} )) {%- endif -%}
{%- endmacro %}
