#import discord.py library and init extensions
import discord
from discord import app_commands
import discord.ext
from texit import latex2image
import yt_dlp

#standard python libraries
import urllib
import re
from numpy import random

queuelist = {}

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

@client.event
async def on_ready():
    print(f'Successfully logged in as {client.user}')
    
@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if message.author.id != 1199215459782893569:
        if client.user.mentioned_in(message) or not message.guild:
            return
        
#interesting commands
@tree.command(
    name="mad",
    description="make people mad"
)
async def mad(interaction):
    await interaction.response.send_message("mad?")

@tree.command(
    name="xanicle",
    description="only for my brudda xanicle"
)
async def xanicle(interaction):
    if interaction.user.id == 738927025405821015:
        await interaction.response.send_message("hello my fellow robot")
    else:
        await interaction.response.send_message("hey! you're not xanicle????")

#youtube
class Song:
    def __init__(self,url:str):
        ydl_opts = {'format': 'bestaudio'}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl: #extract raw url of video
            song_info = ydl.extract_info(url, download=False)
        self.url = song_info["url"]

class Queue:
    def __init__(self,voice_client):
        self.voice_client = voice_client
        self.songlist = []
        self.paused = False
        self.loop = False
    
    async def addsong(self, url:str):
        self.songlist.append(Song(url))
    
    def nextsong(self):
        if self.loop:
            self.songlist.append(self.songlist[0])
        del self.songlist[0]
        if len(self.songlist) > 0:
            ffmpeg_options = {'before_options':'-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5','options': '-vn'}
            self.voice_client.play(discord.FFmpegPCMAudio(self.songlist[0].url,**ffmpeg_options),after= lambda e: self.nextsong())

    async def play(self):
        self.voice_client.stop()
        ffmpeg_options = {'before_options':'-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5','options': '-vn'}
        self.voice_client.play(discord.FFmpegPCMAudio(self.songlist[0].url,**ffmpeg_options),after= lambda e: self.nextsong())

@tree.command(name="queue",description="Searches for and queues a youtube video")
async def queue(interaction,search : str):
    serverid = str(interaction.guild.id)
    try: queuelist[serverid]
    except: queuelist[serverid] = Queue(voice_client=interaction.guild.voice_client)

    search = search.replace(" ", "+")

    #get first result in search
    html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search)
    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

    #send video
    url = "https://www.youtube.com/watch?v=" + video_ids[0]
    await interaction.response.send_message(f"added {url} at position {len(queuelist[serverid].songlist) + 1}")

    await queuelist[serverid].addsong(url)

@tree.command(name="betterqueue",description="Searches for and queues multiple youtube videos. Seperate each search with a comma")
async def betterqueue(interaction,search : str):
    search = search.replace(" ", "+")
    search = search.split(",")
    serverid = str(interaction.guild.id)
    try: queuelist[serverid]
    except: queuelist[serverid] = Queue(voice_client=interaction.guild.voice_client)

    await interaction.response.send_message(f"added {len(search)} songs to queue")

    for x in search:
        #get first result in search
        html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + x)
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

        #send video
        url = "https://www.youtube.com/watch?v=" + video_ids[0]

        await queuelist[serverid].addsong(url)

@tree.command(name = "join", description="make bot join and make people mad")
async def join(interaction):
    serverid = str(interaction.guild.id)
    try: queuelist[serverid]
    except: queuelist[serverid] = Queue(voice_client=interaction.guild.voice_client)

    if interaction.user.voice == None:
        await interaction.response.send_message("You must be in a voice channel to use this command!")
        return

    voice_client = interaction.guild.voice_client
    voiceChannel = interaction.user.voice.channel

    if voice_client and voice_client.is_connected():
        if voice_client.channel != voiceChannel:
            await voice_client.move_to(voiceChannel)
    else:
        voice_client = await voiceChannel.connect()
    voice_client.stop()

    await interaction.response.send_message("Are you mad by any chance?")
    queuelist[serverid].voice_client = interaction.guild.voice_client
    await queuelist[serverid].play()

@tree.command(name="loop",description="toggles whether or not the queue loops")
async def loop(interaction):
    serverid = str(interaction.guild.id)
    try: queuelist[serverid]
    except: queuelist[serverid] = Queue(voice_client=interaction.guild.voice_client)
    
    if queuelist[serverid].loop:
        await interaction.response.send_message("Loop has been disabled")
    else:
        await interaction.response.send_message("Loop has been enabled")
    
    queuelist[serverid].loop = not queuelist[serverid].loop

@tree.command(name="clearqueue",description="Clears the song queue or removes a song from queue")
async def clearqueue(interaction,position : int = None):
    serverid = str(interaction.guild.id)
    try: queuelist[serverid]
    except:
        await interaction.response.send_message("No songs currently in queue")
        return
    if position == None:
        queuelist[serverid].songlist = []
        await interaction.response.send_message("Queue successfully cleared")
    else:
        try:
            del queuelist[serverid].songlist[position-1]
            await interaction.response.send_message("Song removed from queue")
        except:
            await interaction.response.send_message("Invalid position")
            return
    

@tree.command(name="skip",description="skips current song in queue")
async def skip(interaction):
    serverid = str(interaction.guild.id)
    try: queuelist[serverid]
    except:
        await interaction.response.send_message("No songs currently in queue")
        return
    
    del queuelist[serverid].songlist[0]
    interaction.guild.voice_client.stop()
    await interaction.response.send_message("Song skipped")
    queuelist[serverid].voice_client = interaction.guild.voice_client
    await queuelist[serverid].play()

@tree.command(name="leave",description = "disconnects the bot from a voice channel")
async def leave(interaction):
    if interaction.user.id != 563661296361275393: #check if user is Samuel Wu
        for x in client.voice_clients:
            if x.guild == interaction.guild:
                await x.disconnect()
        await interaction.response.send_message("Left all active voice channels!")
    else:
        await interaction.response.send_message("No. Mad?")

@tree.command(name="pause",description="pause or resume a youtube video")
async def pause(interaction):
    vc = interaction.guild.voice_client
    if not vc.is_playing():
        await interaction.response.send_message("Youtube video resumed")
        vc.resume()
    else:
        await interaction.response.send_message("Youtube video paused")
        vc.pause()

@tree.command(name="play",description = "searches for and plays a youtube video")
async def play(interaction,search : str, musicvolume : float = 1.0):

    if interaction.user.voice == None:
        await interaction.response.send_message("You must be in a voice channel to use this command!")
        return

    search = search.replace(" ", "+")

    #get first result in search
    html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search)
    video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())

    #send video
    url = "https://www.youtube.com/watch?v=" + video_ids[0]
    await interaction.response.send_message(url)


    voice_client = interaction.guild.voice_client

    voiceChannel = interaction.user.voice.channel
    if voice_client and voice_client.is_connected():
        if voice_client.channel != voiceChannel:
            await voice_client.move_to(voiceChannel)
    else:
        voice_client = await voiceChannel.connect()
    
    ffmpeg_options = {'before_options':'-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5','options': '-vn'}
    ydl_opts = {'format': 'bestaudio'}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl: #extract raw url of video
        song_info = ydl.extract_info(url, download=False)
    
    #play video and change volume
    source = discord.FFmpegOpusAudio(song_info["url"],**ffmpeg_options)
    voice_client.stop()
    voice_client.play(source)

#math stuff
async def get_question(paper):
    qnum = random.randint(0,11)
    qlist = ["A1","A2","A3","A4","A5","A6","B1","B2","B3","B4","B5","B6"]
    sub1 = f"\\item[{qlist[qnum]}]"
    sub2 = f"\\item[{qlist[qnum+1]}]"
    test_str=paper.replace(sub1,",.!*")
    test_str=test_str.replace(sub2,",.!*")
    re=test_str.split(",.!*")
    res=re[1]
    return res

@tree.command(name="putnam",description="Gets a random putnam problem A1 - B6")
async def putnam(interaction):
    channel = interaction.channel
    with open(file=f"latex/{random.randint(2013,2024)}.tex",mode="r") as f:
        paper = f.read()
    question = await get_question(paper)
    await latex2image(question)
    await interaction.response.send_message(f"mald")
    await channel.send(file=discord.File("image.png"))

    

#owner only
@tree.command(name='sync', description='Owner only')
async def sync(interaction):
    if interaction.user.id == "OWNER ID HERE":
        await tree.sync()
        await interaction.response.send_message('Command tree synced')
    else:
        await interaction.response.send_message('this command is only for the owner of this bot')

client.run('BOT TOKEN')