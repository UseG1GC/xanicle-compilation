execute unless predicate xanicle:offhand if score @s pop_counter < @s bot_totems run function xanicle:totem/retotem1
execute if score @s botState matches 3 unless predicate xanicle:mainhand if score @s pop_counter < @s bot_totems run function xanicle:totem/retotem
execute if score @s totem_cd >= @s totem_cooldown run scoreboard players set @s[scores={botState=3}] botState 1
scoreboard players add @a[tag=bot,scores={totem_cd=..300}] totem_cd 1
execute if score @s botState matches 3 run player xaniclebot stop