scoreboard players set xaniclebot botState 1
scoreboard players set @a dead 0
function xanicle:resetmap
tp @a[tag=bot] 15 51 0
tp @a[tag=!bot,gamemode=!spectator] -15 51 0 facing 0 51 0
scoreboard players set xaniclebot pop_counter 0
gamemode survival @a
scoreboard players set @a gameState 1