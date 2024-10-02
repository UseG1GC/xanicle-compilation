execute unless entity @a[tag=bot,distance=..3] run kill @s
execute positioned ~ ~1 ~ if entity @a[distance=..0.7] run kill @s
execute if entity @a[distance=..1] run kill @s
execute unless entity @a[tag=!bot,gamemode=!spectator,distance=..5] run kill @s