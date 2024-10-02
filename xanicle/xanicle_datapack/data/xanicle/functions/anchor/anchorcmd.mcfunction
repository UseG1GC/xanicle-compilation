setblock ~ ~ ~ air
execute unless block ~ ~-1 ~ air if entity @a[tag=bot,distance=..3] unless entity @a[distance=..0.5,gamemode=!spectator] run summon marker ~ ~ ~ {Tags:["anchor","loc1"]}