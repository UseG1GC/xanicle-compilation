player xaniclebot hotbar 9
scoreboard players set @s botState 3
execute unless predicate xanicle:mainhand run item replace entity @s weapon.mainhand with totem_of_undying
scoreboard players add @s pop_counter 1
scoreboard players set @a[tag=bot] totem_cd 0