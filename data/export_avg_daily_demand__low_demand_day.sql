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
        round(sum(Kg)/180) as 'Kg', -- quick & dirty daily avg, 180 working days in 9 months (2023)
        ProductType
    from sales_2023
    group by ClientID, ProductType
) a

where a.Lat>0 and a.Lat<10 -- hack tofilter incomplete/dirty records
AND a.ClientID % 10 = 0 -- select approximately 10% of clients
and a.Kg >= 0;
