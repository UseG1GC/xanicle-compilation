function xanicle:look
execute if score @s swing_cd matches 9 run player xaniclebot jump
execute if score @s swing_cd matches 8.. run player xanicleBot stop
execute if score @s swing_cd matches ..8 if entity @p[tag=!bot,gamemode=!spectator,distance=3.1..] run player xanicleBot move forward
execute if score @s swing_cd matches 0 if entity @p[distance=..3,tag=!bot,gamemode=!spectator] run function xanicle:axe/swordhit
execute if score @s swing_cd matches 3.. if entity @p[tag=!bot,gamemode=!spectator,distance=..3.1] run player xanicleBot move backward