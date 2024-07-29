player xaniclebot spawn in creative
tag xaniclebot add bot
tag xaniclebot add refill

scoreboard players set xaniclebot swing_cd 0
scoreboard players set xaniclebot obby_cd 0
scoreboard players set xaniclebot crystal_cd 0
scoreboard players set xaniclebot totem_cd 0
scoreboard players set xaniclebot anchor_cd 0
scoreboard players set xaniclebot bot_target 1
scoreboard players set xaniclebot botState -1

give xaniclebot totem_of_undying
item replace entity xaniclebot weapon.offhand with totem_of_undying
scoreboard players set xaniclebot pop_counter 0