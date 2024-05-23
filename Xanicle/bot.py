import discord
from discord import app_commands
import discord.ext
import os
from dotenv import load_dotenv
from fish import Fish
import json 
import yt_dlp

with open('themes.json', 'r') as file:
    themes = json.load(file) 

load_dotenv()

import pyttsx3

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)
timeout = None
queues = {}

fish = Fish("http://localhost/api/chat", "You are a fish.")

Token = os.getenv('DISCORD_TOKEN')

voice = pyttsx3.init()
voice.setProperty('rate',150)

@client.event
async def on_ready():
    print(f'Successfully logged in as {client.user}')

@client.event
async def on_message(message):
    author = message.author.mention
    
    if "<@417216848720035862>" in message.content:
        await message.delete()
        await message.channel.send(f"Don't disturb my fellow 🐟! {author}")
    
    if client.user.mentioned_in(message):
        payload = message.content.replace("<@1199290199985901589>", "")
        reply = fish.generate(payload)

        if payload != "":
            await message.reply(reply)
        else:
            await message.reply("Blop")

@client.event
async def on_voice_state_update(member, before, after):
    global themes
    if before.channel != after.channel:
        print(f"{member} moved from {before.channel} to {after.channel}")

    voice_client = discord.utils.get(client.voice_clients, guild=member.guild)

    if after.channel and (not voice_client or voice_client.channel != after.channel):
        if voice_client and voice_client.is_connected():
            print(f"Moving to {after.channel}")
            await voice_client.move_to(after.channel)
        else:
            print(f"Connecting to {after.channel}")
            await after.channel.connect()

    if not after.channel and voice_client and len(voice_client.channel.members) == 1:
        print(f"Disconnecting from {voice_client.channel} as it is empty.")
        await voice_client.disconnect()

    if str(member.id) in themes and voice_client and after.channel != None:
        url = themes[str(member.id)]
        
        ffmpeg_options = {'options': '-vn'}
        ydl_opts = {'format': 'bestaudio'}

        with yt_dlp.YoutubeDL(ydl_opts) as ydl: #extract raw url of video
            song_info = ydl.extract_info(url, download=False)

        source = discord.FFmpegPCMAudio(song_info["url"], executable="ffmpeg", options=ffmpeg_options)
        if voice_client.is_playing():
            voice_client.pause()
        voice_client.play(source)
        print("PLAYING")
        
        with open('themes.json', 'r') as file:
            themes = json.load(file) 
    elif(after.channel != None):
        print(f"{member} is a USER WITHOUT THEME THAT HAS JOINED")


@tree.command(name='sync', description='Owner only')
async def sync(interaction):
    if interaction.user.id == 417216848720035862:
        await tree.sync()
        await interaction.response.send_message('Command tree synced')
    else:
        await interaction.response.send_message('this command is only for the owner of this bot')

@tree.command(name='echo', description='Repeats what you say')
async def echo(interaction: discord.Interaction, message: str):
    await interaction.response.send_message(message)

@tree.command(name='ping', description='Pong!')
async def ping(interaction):
    await interaction.response.send_message(f"Pong! {round(interaction.client.latency * 1000)}ms")

@tree.command(name='fish', description='fish')
async def fish(interaction):
    await interaction.response.send_message('🐟🐟🐟')


@tree.command(name="enter",description = "join the vc")
async def enter(interaction):

    if interaction.user.voice == None:
        await interaction.response.send_message("You must be in a voice channel to use this command!")
        return

    voice = discord.utils.get(client.voice_clients, guild=interaction.guild)

    voiceChannel = interaction.user.voice.channel
    await voiceChannel.connect()
    await interaction.response.send_message('A wild 🐟 has appeared')


@tree.command(name="exit", description="make 🐟 disappear")
async def exit(interaction: discord.Interaction):
    voice_client = interaction.guild.voice_client
    if interaction.user.voice == None:
        await interaction.response.send_message('I am not connected to a voice channel')
        return
    
    if voice_client.is_connected():
        await voice_client.disconnect()
        await interaction.response.send_message('The 🐟 has disappeared')
    

@tree.command(name="speak", description="make 🐟 speak")
async def search(interaction: discord.Interaction, speech: str, volume: float):
    
    if interaction.user.voice is None:
        await interaction.response.send_message("You are not connected to a voice channel.")
        return
    await interaction.response.defer()
    voice_channel = interaction.user.voice.channel
    voice_client = interaction.guild.voice_client
    

    if voice_client and voice_client.is_connected():
        if voice_client.channel != voice_channel:
            await voice_client.move_to(voice_channel)
    else:
        voice_client = await voice_channel.connect()
    
    voice.setProperty('volume', volume)
    voice.save_to_file(speech, 'speech.mp3')
    voice.runAndWait()
    

    ffmpeg_options = {'options': '-vn'}
    source = discord.FFmpegPCMAudio('speech.mp3', executable="ffmpeg", options=ffmpeg_options)
    if not voice_client.is_playing():
        voice_client.play(source, after=lambda e: os.remove('speech.mp3'))
        await interaction.followup.send("The 🐟 has spoken.")
    else:
        await interaction.followup.send("Please wait until the 🐟 has spoken!")


@tree.command(name="play",description = "make 🐟 play music")
async def music(interaction: discord.Interaction, url : str):

    if interaction.user.voice is None:
        await interaction.response.send_message("You are not connected to a voice channel.")
        return
    await interaction.response.defer()
    voice_channel = interaction.user.voice.channel
    voice_client = interaction.guild.voice_client

    if voice_client and voice_client.is_connected():
        if voice_client.channel != voice_channel:
            await voice_client.move_to(voice_channel)
    else:
        voice_client = await voice_channel.connect()

    ffmpeg_options = {'options': '-vn'}
    ydl_opts = {'format': 'bestaudio'}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl: #extract raw url of video
        song_info = ydl.extract_info(url, download=False)

    source = discord.FFmpegPCMAudio(song_info["url"], executable="ffmpeg", options=ffmpeg_options)
    author = interaction.user.mention

    if not voice_client.is_playing():
        voice_client.play(source, after=lambda e: os.remove("music.mp3"))
        await interaction.followup.send(url)
    else:
        await interaction.followup.send(f"Please wait for the 🐟 to finish! {author}")

@tree.command(name="pause" ,description="Pauses the 🐟.")
async def pause(interaction):
    if interaction.user.voice is None:
        await interaction.response.send_message("You are not connected to a voice channel.")
        return

    
    voice_client = interaction.guild.voice_client
    author = interaction.user.mention

    voice_client.pause()
    await interaction.response.send_message(f"{author} has paused 🐟")
    

@tree.command(name="resume" ,description="Resumes the 🐟.")
async def pause(interaction):
    if interaction.user.voice is None:
        await interaction.response.send_message("You are not connected to a voice channel.")
        return

    
    voice_client = interaction.guild.voice_client
    author = interaction.user.mention

    voice_client.resume()
    await interaction.response.send_message(f"{author} has resumed 🐟")

@tree.command(name="volume" ,description="Changes the 🐟's volume.")
async def volume(interaction: discord.Interaction, volume : float ):
    if interaction.user.voice is None:
        await interaction.response.send_message("You are not connected to a voice channel.")
        return
    
    vc = interaction.guild.voice_client
    vc.source = discord.PCMVolumeTransformer(vc.source)
    vc.source.volume = volume

    author = interaction.user.mention

    await interaction.response.send_message(f"{author} has changed 🐟's volume to {volume * 100}%")

@tree.command(name = "theme", description="Set a background music for 🐟 to play when you join")
async def theme(interaction: discord.Interaction, url : str):
    global themes
    author = interaction.user.mention
    something = interaction.user.id
    themes[str(something)] = url
    with open("themes.json", "w") as outfile:
        json.dump(themes, outfile)
    await interaction.response.send_message(f"🐟 has successfully changed {author}'s theme")

client.run(Token)
