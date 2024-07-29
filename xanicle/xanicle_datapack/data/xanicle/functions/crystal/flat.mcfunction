execute at @p[tag=!bot,gamemode=!spectator] run function xanicle:marker/marklow
execute unless entity @e[tag=lowpos,distance=..3.3] if score @p[tag=bot] swing_cd matches 0 run function xanicle:pearl