execute as @a[tag=bot] on attacker at @s if entity @s[tag=!bot,gamemode=!spectator,type=player] if score @p[tag=bot] flat_cd >= @p[tag=bot] flat_cooldown run function xanicle:dtap/pflash
