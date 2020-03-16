
select * from bomen.xy_bomen
create table kerngis.relevante_vlakken as(
select objectteks from kerngis.groen_vlakken where objectteks in ('bomengroep', 'Bomen', 'Boomgroep', 'Bomengroep',
																		   'bmn en Struiken', 'Loofbos', 'Bos (vervalt)',
																		   'Bomen en struiken', 'Bmn en Struiken', 'Bos', 'bmn',
																		   'bomen en Struiken', 'Boomn groep', 'bos', 'bmn en strkn')
);

create table kerngis.relevante_punten as (
select * from kerngis.groen_punten where objectteks in ('Boom', 'Solitaire boom', 'BooM')
);

create table kerngis.relevante_lijnen as (
select * from kerngis.groen_lijnen where soort in ('BR')
);

