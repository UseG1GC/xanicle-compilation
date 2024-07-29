execute at @e[tag=explosion,limit=1,sort=nearest] run function xanicle:anchor/detonate
execute at @e[tag=glowstone,limit=1,sort=nearest] unless entity @e[tag=explosion] run function xanicle:anchor/placeglowstone
execute at @e[tag=anchor,limit=1,sort=nearest] unless entity @e[tag=glowstone] unless entity @e[tag=explosion] if score xaniclebot anchor_cd >= xaniclebot anchor_cooldown run function xanicle:anchor/placeanchor
execute unless entity @e[tag=loc1,type=marker] run function xanicle:anchor/spawnmarker
