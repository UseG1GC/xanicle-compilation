setblock ~ ~ ~ obsidian
execute positioned ~ ~1 ~ unless entity @a[tag=!bot,gamemode=!spectator,distance=..1] run kill @e[type=marker,distance=..0.1]