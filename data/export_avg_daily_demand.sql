SELECT
    a.ClientID,
    a.Lat,
    a.Lon,
    CAST(a.Kg as INTEGER) as 'Kg',
    a.ProductType
from (
    select
        ClientID,
        Lat,
        Lon,
        round(sum(Kg)/9/30) as 'Kg', -- quick & dirty daily avg
        ProductType
    from sales_2023
    group by ClientID, ProductType
) a

where a.Lat>0 and a.Lat<10 -- hack tofilter incomplete/dirty records
and a.Kg >= 0;
