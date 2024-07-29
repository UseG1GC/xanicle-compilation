setblock ~ ~ ~ respawn_anchor[charges=1]
execute if entity @a[tag=bot,distance=..3] run summon marker ~ ~ ~ {Tags:["explosion","loc1"]}