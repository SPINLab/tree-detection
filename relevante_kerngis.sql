drop table if exists kerngis.relevante_vlakken;
create table kerngis.relevante_vlakken as(
select * from kerngis.groen_vlakken where objectteks in ('bomengroep', 'Bomen', 'Boomgroep', 'Bomengroep', 
																		   'bmn en Struiken', 'Loofbos', 'Bos (vervalt)', 
																		   'Bomen en struiken', 'Bmn en Struiken', 'Bos', 'bmn', 
																		   'bomen en Struiken', 'Boomn groep', 'bos', 'bmn en strkn')
);

drop table if exists kerngis.relevante_punten;
create table kerngis.relevante_punten as (
select * from kerngis.groen_punten where objectteks in ('Boom', 'Solitaire boom', 'BooM')
);

drop table if exists kerngis.relevante_lijnen;
create table kerngis.relevante_lijnen as (
select * from kerngis.groen_lijnen where soort in ('BR')
);