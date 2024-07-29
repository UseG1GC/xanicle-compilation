kill @e[tag=target]
scoreboard players set @a[tag=bot] botState -2
player xanicleBot hotbar 4
function xanicle:looklow
player xanicleBot use once
scoreboard players set @a[tag=bot] bot_target 1
scoreboard players set @a[tag=bot] swing_cd 20
scoreboard players set @a[tag=bot] botState 1