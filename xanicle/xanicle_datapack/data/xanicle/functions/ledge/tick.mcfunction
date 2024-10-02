execute as @a[tag=bot] at @s run damage @e[tag=target,limit=1,sort=nearest,distance=..4] 1 player_attack by @s from @s
execute if score @p[tag=bot] crystal_cd >= @p[tag=bot] crystal_cooldown run function xanicle:crystal/spawncrystal
execute if score @p[tag=bot] obby_cd >= @p[tag=bot] obsidian_cooldown run function xanicle:crystal/spawnobby

scoreboard players add @a[tag=bot,scores={obby_cd=..300}] obby_cd 1
scoreboard players add @a[tag=bot,scores={crystal_cd=..300}] crystal_cd 1