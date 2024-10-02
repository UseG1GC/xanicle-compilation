function xanicle:look
player xaniclebot stop
player xaniclebot attack once
execute if entity @p[tag=!bot,gamemode=!spectator,scores={shield=1..}] run function xanicle:axe/stun
scoreboard players set xaniclebot swing_cd 11
player xaniclebot use continuous