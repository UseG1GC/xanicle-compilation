execute as @a[tag=bot] at @s if entity @p[tag=!bot,gamemode=!spectator,distance=3..5] unless entity @e[tag=loc,distance=..3] at @p[tag=!bot,gamemode=!spectator] run function xanicle:anchor/place
scoreboard players add @a[tag=bot,scores={anchor_cd=..300}] anchor_cd 1
kill @e[tag=loc1,type=marker]