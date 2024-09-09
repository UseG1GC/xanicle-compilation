scoreboard objectives add botState dummy
scoreboard objectives add swing_cd dummy
scoreboard objectives add bot_target dummy

scoreboard objectives add timer dummy
scoreboard objectives add obby_cd dummy
scoreboard objectives add crystal_cd dummy
scoreboard objectives add crystal_cooldown dummy
scoreboard objectives add obsidian_cooldown dummy
scoreboard objectives add flat_cooldown dummy
scoreboard objectives add flat_cd dummy
scoreboard objectives add totem_cd dummy
scoreboard objectives add totem_cooldown dummy
scoreboard objectives add anchor_cd dummy
scoreboard objectives add anchor_cooldown dummy

scoreboard objectives add bot_totems dummy
scoreboard objectives add totem_cd dummy
scoreboard objectives add totem_cooldown dummy
scoreboard objectives add pop_counter dummy
scoreboard objectives add hole dummy
scoreboard objectives add anchor dummy
scoreboard objectives add gameState dummy

scoreboard objectives add dead deathCount
scoreboard objectives add hp health
scoreboard objectives add shield minecraft.custom:minecraft.damage_blocked_by_shield

scoreboard players set xaniclebot bot_totems 15
function xanicle:presets/normal
function xanicle:enablehole

gamerule commandBlockOutput false