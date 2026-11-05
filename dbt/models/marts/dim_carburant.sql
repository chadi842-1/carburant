select * from (
    values
        (1, 'Gazole', 'diesel'),
        (2, 'SP95',   'essence'),
        (3, 'E85',    'bioéthanol'),
        (4, 'GPLc',   'gaz'),
        (5, 'E10',    'essence'),
        (6, 'SP98',   'essence')
) as t(carburant_id, carburant, famille)
