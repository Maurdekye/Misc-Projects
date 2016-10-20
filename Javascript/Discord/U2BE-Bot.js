const discord_api = require("discord.js");
const ytdl = require('ytdl-core');
const fs = require('fs');

var tokens = JSON.parse(fs.readFileSync("botinfo.json"));
var playlist = [];
var playing = false;
var dispatch;

const bot = new discord_api.Client();

function recurPrintPlaylistNames(channel, callbefore=()=>{}, names=[], index=1) {
  if (index === playlist.length) {
    ytdl.getInfo(playlist[0], (err, info) => {
      var s = "```Current video: " + info.title + "\nComing up:\n";
      for (var i=0;i<names.length;i++) {
        s = s + "   " + (i + 1) + ". " + names[i] + "\n"
      }
      s = s + "```";
      channel.sendMessage(s);
    });
  } else {
    ytdl.getInfo(playlist[index], (err, info) => {
      recurPrintPlaylistNames(channel, callbefore, names.concat([info.title]), index + 1);
    });
  }
}

function timestamp() {
  var ctime = new Date();
  var hrs = ctime.getUTCHours();
  if (hrs < 10) hrs = "0" + hrs;
  var mns = ctime.getUTCMinutes();
  if (mns < 10) mns = "0" + mns;
  var sec = ctime.getUTCSeconds();
  if (sec < 10) sec = "0" + sec;
  return "[" + hrs + ":" + mns + ":" + sec + "]";
}

function log(text) {
  console.log(timestamp() + " " + text);
}

function recurExhaustPlaylist(vchannel, tchannel, connection) {
  if (playlist.length === 0) {
    playing = false;
    vchannel.leave();
  } else {
    playing = true;
    ytdl(playlist[0], (err, info) => {
      tchannel.sendMessage("Now playing: `" + info.title + "`");
      log("Started playing new song: '" + info.title + "'");
    });
    dispatch = connection.playStream(ytdl(playlist[0], {audioonly: true}));
    dispatch.setVolumeLogarithmic(0.3);
    dispatch.on('end', () => {
      if (playing) {
        playlist.shift();
        recurExhaustPlaylist(vchannel, tchannel, connection);
      } else {
        log("Stopped playing.");
        vchannel.leave();
      }
    });
  }
}

bot.on("message", msg => {
  var text = msg.content;
  var args = text.split(" ").filter( l => l.length > 0 );
  if (args.length === 0) {
    return;
  }

  if (args[0] === "!list") {
    if (args.length == 1) {
      if (playlist.length == 0) {
        msg.channel.sendMessage("No videos in playlist; type `!add <video url>` to add a video.");
      } else if (playlist.length == 1) {
        ytdl.getInfo(playlist[0], (err, info) => {
          msg.channel.sendMessage("Current video: `" + info.title + "`");
        });
      } else {
        recurPrintPlaylistNames(msg.channel);
      }
    } else if (args[1] === "clear") {
      playlist = [];
      msg.channel.sendMessage("Cleared playlist.");
      log("Cleared playlist.");
    }
  } else if (args[0] === "!add") {
    if (args.length == 1) {
        msg.channel.sendMessage("Provide a YouTube video url; `!add <video url>`");
    } else {
      ytdl.getInfo(args[1], (err, info) => {
        if (info) {
          playlist.push(args[1]);
          msg.channel.sendMessage("Added `" + info.title + "` to playlist");
          log("Added new song to playlist: '" + info.title + "'");
        } else {
          msg.channel.sendMessage("`" + args[1] + "` is not a valid YouTube video url.");
        }
      });
    }
  } else if (args[0] === "!play") {
    if (playlist.length === 0) {
      msg.channel.sendMessage("No videos in playlist; type `!add <video url>` to add a video.");
    } else if (!msg.member.voiceChannel) {
      msg.channel.sendMessage("Join a voice channel before using `!play`");
    } else {
      msg.member.voiceChannel.join().then( c => recurExhaustPlaylist(msg.member.voiceChannel, msg.channel, c));
    }
  } else if (args[0] === "!stop") {
    if (!playing) {
      msg.channel.sendMessage("Not currently playing in a channel.");
    } else {
      playing = false;
      dispatch.end();
    }
  } else if (args[0] === "!skip") {
    if (playlist.length === 0) {
      msg.channel.sendMessage("No videos in playlist.");
    } else {
      ytdl(playlist[0], (err, inf) => {
        msg.channel.sendMessage("Skipped video `" + inf.title + "`");
        log("Skipped current song, '" + info.title + "'");
      });
      if (playing) {
        dispatch.end();
      } else {
        playlist.shift();
      }
    }
  }
});

log("Connecting to discord");
bot.login(tokens.discord_api_token);