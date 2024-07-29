execute if entity @a[tag=bot,scores={hole=1}] run fill 0 50 0 0 -40 0 command_block{auto:1b,Command:"fill ~-75 ~ ~-75 ~75 ~ ~75 stone"}
execute if entity @a[tag=bot,scores={hole=1}] run fill 0 -41 0 0 -50 0 command_block{auto:1b,Command:"fill ~-75 ~ ~-75 ~75 ~ ~75 barrier"}
execute if entity @a[tag=bot,scores={hole=0}] run fill 0 50 0 0 40 0 command_block{auto:1b,Command:"fill ~-75 ~ ~-75 ~75 ~ ~75 netherite_block"}
fill 0 51 0 0 120 0 command_block{auto:1b,Command:"fill ~-75 ~ ~-75 ~75 ~ ~75 air"}