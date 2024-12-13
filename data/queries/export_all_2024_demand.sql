SELECT
    Date,
    ClientID,
    Lat,
    Lon,
    CAST(SUM(Kg) as INTEGER) as Kg,
    ProductType
FROM sales
WHERE Lat BETWEEN 4.3333 AND 4.9167 
    AND Lon BETWEEN -74.3500 AND -73.9167  -- Bogota Metro Area bounding box
    AND Date LIKE '2024%'  -- Filter for all days in 2024
GROUP BY Date, ClientID, ProductType, Lat, Lon
HAVING SUM(Kg) >= 2  -- Remove small demand at client level
    AND SUM(Kg) <= 4500  -- No outliers at client level 