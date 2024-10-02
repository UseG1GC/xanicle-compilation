setblock ~ ~ ~ obsidian
execute if block ~ ~1 ~ air run summon marker ~ ~ ~ {Tags:["obby","loc"]}
execute as @e[tag=obby,limit=1,sort=nearest] at @s run function xanicle:marker/check_loc