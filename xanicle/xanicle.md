# XANICLEBOT INFO
xaniclebot is a crystal pvp bot, utilizing carpet's fakeplayer functionality\
Xaniclebot has the ability to:
 - Play flat (no anchors)
 - Play hole (with anchors)
 - Play hole (no anchors)
## General Crystal Information
Xaniclebot detects and marks the location of obsidian present underneth the target\
Xaniclebot also detects and marks the location of air blocks where obsidian could be placed\
2 variables are used:
 - Crystal cooldown, which controls the speed at which the bot places end crystals
 - Obsidian cooldown, which controls the time before xaniclebot can place another obsidian block
xaniclebot uses a similar method to place and detonate respawn anchors
 - Anchor cooldown controls the time before xaniclebot places another anchor
In order to manually control these variables, use:
    /scoreboard players set xaniclebot crystal_cooldown <value>
    /scoreboard players set xaniclebot obsidian_cooldown <value>
    /scoreboard players set xaniclebot anchor_cooldown <value>
crystal_cd, obsidian_cd and anchor_cd are used as timers
## Bot State / Bot target
Xaniclebot uses a variable (bot state) and a flat (bot target) to control bot targeting and behaviour\
Bot States:
0. Sword PvP - only allows the bot to move forward and attack
1. Normal / idle - allows the bot to place obsidian, crystal and anchors, as well as to pearl and pearl flash
2. Hitcrystaling - allows the bot to hit its opponent. State only set when the bot is able to attack the opponent. State is set back to 1 after a short delay
3. Retoteming - bot does nothing except retotem. State set back to 3 after a short cooldown.
When set to a value other than 0 ~ 3, xaniclebot does nothing.\
\
Bot Targets:
0. targeting nearest obsidian block or possible location for obsidian
1. Targeting player
