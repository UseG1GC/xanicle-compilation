execute if score @s botState matches 4 run function xanicle:axe/aggro
execute if score @s botState matches 5 run function xanicle:smp/camcal
execute if score @s hp < @p[tag=!bot,gamemode=!spectator] hp if score @s hp matches ..12 run scoreboard players set @s botState 5
execute if score @s hp matches 14.. run scoreboard players set @s botState 4