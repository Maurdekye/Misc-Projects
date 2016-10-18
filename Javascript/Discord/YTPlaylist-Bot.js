const discord_api = require("discord.js");
const ytdl = require('ytdl-core');
var data = require("Bot-Info.json");
const bot = new discord_api.Client();

var playlist = [] 

function recurPrintPlaylistNames(channel, callbefore=()=>{}, names=[], index=0) {
  if (index === playlist.length) {
    var s = "Coming up:\n";
    for (var i=0;i<names.length;i++) {
      s = s + "`  " + (i + 1) + ". " + names[i] + "`\n"
    }
    callbefore();
    bot.sendMessage(channel, s);
  } else {
    ytdl.getInfo(playlist[index], (err, info) => {
      recurPrintPlaylistNames(channel, callbefore, names + [info.title], index + 1);
    });
  }
}

function recurExhaustPlaylist(vchannel, connection) {
  if (playlist.length === 0) {
    vchannel.leave();
  } else {
    const dispatch = connection.playStream(ytdl(playlist[0], {audioonly: true}));
    dispatch.on('end', () => {
      if (!vchannel.guild.voiceChannel) {
        playlist.shift();
        recurExhaustPlaylist(vchannel, connection);
      }
    });
  }
}

bot.on("message", msg => {
  var text = msg.content;
  var args = text.split(" ").filter( l => l.length > 0 );
  if (text.startsWith("!list")) {
    if (args.length == 0) {
      if (playlist.length == 0) {
        bot.sendMessage(msg.channel, "No videos in playlist; type `!add <video url>` to add a video.");
      } else if (playlist.length == 1) {
        ytdl.getInfo(playlist[0], (err, info) => {
          bot.sendMessage(msg.channel, "Now playing: `" + info.title + "`");
        });
      } else {
        recurPrintPlaylistNames(msg.channel, callbefore=() => {
          bot.sendMessage(msg.channel, "Now playing: `" + info.title + "`");
        });
      }
    } else if (args[1] === "clear") {
      playlist = [];
      bot.sendMessage(msg.channel, "Cleared playlist.");
    }
  } else if (text.startsWith("!add")) {
    if (args.length == 1) {
        bot.sendMessage(msg.channel, "Provide a YouTube video url; `!add <video url>`");
    } else {
      ytdl.getInfo(args[1], (err, info) => {
        if (info) {
          playlist.push(args[1]);
          bot.sendMessage(msg.channel, "Added `" + info.title + "` to playlist");
        } else {
          bot.sendMessage(msg.channel, "`" + args[1] + "` is not a valid YouTube video url.");
        }
      });
    }
  } else if (text.startsWith("!play")) {
    if (!msg.author.voiceChannel) {
      bot.sendMessage(msg.channel, "Join a voice channel before using `!play`");
    } else {
      msg.author.voiceChannel.join().then( c => recurExhaustPlaylist(msg.author.voiceChannel, c));
    }
  }
});

console.log("Connecting to discord..")
bot.login(data.discord.api_token);