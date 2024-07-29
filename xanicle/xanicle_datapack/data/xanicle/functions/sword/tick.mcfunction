function xanicle:look
execute if score @s swing_cd matches 8.. run player xanicleBot stop
execute if score @s swing_cd matches ..7 run player xanicleBot move forward
execute if score @s swing_cd matches 0 if entity @p[distance=..3,tag=!bot,gamemode=!spectator] run function xanicle:sword/hit
execute if score @s swing_cd matches 3.. if entity @p[tag=!bot,gamemode=!spectator,distance=..3.1] run player xanicleBot move backward
execute if entity @s[nbt={HurtTime:10s}] run player xaniclebot jump