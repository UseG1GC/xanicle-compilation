execute as @a[tag=bot,scores={botState=1..2}] at @s if score @s flat_cd >= @s flat_cooldown unless entity @a[tag=!bot,gamemode=!spectator,distance=..3] run function xanicle:crystal/flat
scoreboard players add @a[tag=bot,scores={flat_cd=..300}] flat_cd 1
player xaniclebot move forward
execute as @a[tag=bot,scores={bot_target=0}] at @s unless entity @e[type=marker,distance=..3,tag=loc] run scoreboard players set @s bot_target 1
execute as @a[tag=bot] at @s if entity @e[type=marker,tag=loc,distance=..3] run scoreboard players set @s bot_target 0

execute if entity @a[tag=bot,scores={bot_target=1}] unless entity @e[distance=..3,tag=obby] unless entity @e[distance=..3,tag=obby_pos] run function xanicle:look
execute as @a[tag=bot] at @s unless entity @e[distance=..3,tag=obby] at @p[tag=!bot,gamemode=!spectator] at @e[tag=obby_pos,limit=1,sort=nearest] run player xaniclebot look at ~ ~ ~
execute as @a[tag=bot] at @s unless entity @e[distance=..3,tag=obby] at @p[tag=!bot,gamemode=!spectator] at @e[tag=obby_pos,limit=1,sort=nearest] run player xaniclebot hotbar 2
execute at @a[tag=bot] at @p[tag=!bot,gamemode=!spectator] at @e[tag=obby,limit=1,sort=nearest] positioned ~ ~0.5 ~ run player xaniclebot look at ~ ~ ~
execute at @a[tag=bot] at @p[tag=!bot,gamemode=!spectator] at @e[tag=obby,limit=1,sort=nearest] run player xaniclebot hotbar 3
execute at @a[tag=bot] at @s if entity @a[tag=!bot,gamemode=!spectator,predicate=shield] run function xanicle:axe/disable
scoreboard players set @a[tag=bot,scores={bot_target=0,botState=2}] botState 1