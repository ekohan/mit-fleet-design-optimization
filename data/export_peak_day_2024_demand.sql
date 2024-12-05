SELECT
    Date,
    ClientID,
    Lat,
    Lon,
    CAST(SUM(Kg) as INTEGER) as Kg,
    ProductType
FROM sales_2024
WHERE Lat BETWEEN 4.3333 AND 4.9167 
    AND Lon BETWEEN -74.3500 AND -73.9167  -- Bogota Metro Area bounding box
    AND Kg >= 0  -- Has demand
    AND Date = '2024-07-18'  -- Filter for specific day
GROUP BY Date, ClientID, ProductType, Lat, Lon

