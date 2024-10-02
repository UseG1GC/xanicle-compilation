setblock ~ ~ ~ air
execute unless block ~ ~-1 ~ air if block ~ ~1 ~ air run summon marker ~ ~ ~ {Tags:["obby_pos","loc"]}
execute as @e[tag=obby_pos,limit=1,sort=nearest] at @s run function xanicle:marker/check_loc