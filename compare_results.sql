ALTER TABLE results.km_bomen ADD COLUMN correct BOOLEAN;
UPDATE results.km_bomen res
SET correct = 'f';
UPDATE results.km_bomen res
SET correct = 't'
FROM kerngis.relevante_gepold kern
WHERE ST_Intersects(kern.geom, res.geom) AND
ST_Area(ST_Intersection(res.geom, kern.geom)) / ST_Area(res.geom) > 0.3;

-- select where mydata is touched
ALTER TABLE results.km_bomen ADD COLUMN correct BOOLEAN;
UPDATE results.km_bomen res
SET correct_mydata = 'f';
UPDATE results.km_bomen res
SET correct_mydata = 't'
FROM kerngis.relevante_gepold kern
WHERE ST_Intersects(kern.geom, res.geom) AND
ST_Area(ST_Intersection(res.geom, kern.geom)) / ST_Area(res.geom) > 0.3;

-- kerngis
select count(*)
from results.km_bomen
where correct

select 568/ 908.0
-- total: 908
-- correct: 568
-- ratio: 62.55

-- my_data
select count(*)
from results.km_bomen
where km_bomen.correct_mydata

select 727.0 / 908
-- total: 908
-- correct: 727
-- ratio: 80.07

