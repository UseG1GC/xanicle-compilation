# BotStates: 0 - sword, 1 - crystal flat, 2 - hitcrystalling, 3 - retoteming, 4 - smp aggro, 5 - smp healing

execute if score @p[tag=bot] botState matches 1 run function xanicle:ledge/tick
execute if score @p[tag=bot] botState matches 1 run function xanicle:dtap/tick
execute if score @p[tag=bot] botState matches 1..2 run function xanicle:crystal/tick
execute as @a[tag=bot,scores={anchor=1,botState=1..2}] at @s run function xanicle:anchor/tick
execute as @p[tag=bot,scores={botState=2,bot_target=1}] at @s if entity @p[tag=!bot,gamemode=!spectator,distance=..3] run function xanicle:dtap/hit

execute as @p[tag=bot,scores={botState=0}] at @s run function xanicle:sword/tick

execute as @p[tag=bot,scores={botState=2}] at @s unless entity @p[tag=!bot,gamemode=!spectator,distance=..4] run scoreboard players set @s botState 1
execute as @p[tag=bot] at @s run function xanicle:marker/tick

scoreboard players remove @a[scores={swing_cd=1..}] swing_cd 1
scoreboard players remove @a[scores={timer=1..}] timer 1
function xanicle:crystal/refill

execute as @a[tag=bot] at @s if entity @p[tag=!bot,gamemode=!spectator,distance=..3,nbt={HurtTime:0s}] if score @s flat_cd >= @s flat_cooldown run scoreboard players set @a[tag=bot] botState 2
scoreboard players set @a[tag=bot,scores={botState=2}] flat_cd 0

kill @e[type=endermite]
effect give @a[tag=refill] saturation infinite 255 true
execute as @a[tag=bot] at @s run function xanicle:totem/tick

# execute unless entity @a[tag=bot] positioned 1 130 8 run function xanicle:spawnbot
execute at @a[tag=bot] run tp @a[tag=bot,scores={botState=-1}] 1 130 8
function xanicle:end/tick
effect give @a[tag=bot,scores={botState=-1}] resistance 1 255 true