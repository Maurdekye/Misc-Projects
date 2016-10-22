const discord_api = require("discord.js");
const ytdl = require('ytdl-core');
const fs = require('fs');
const https = require('https');
const urlUtils = require('url');

// global vars

const maxmessagelength = 45;

var tokens = JSON.parse(fs.readFileSync("botinfo.json"));
var queues = {};
var playing = false;
var dispatch;

const bot = new discord_api.Client();

// logging functions

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

// url helpers

function makeQueryString(obj) {
  var qlst = [];
  for (var prop in obj) {
    qlst.push(prop + "=" + obj[prop]);
  }
  return "?" + qlst.join("&");
}

function getQueryStringObject(url) {
  var u = urlUtils.parse(url);
  if (!u.query) return {};
  var obj = {};
  var queries = u.query.split("&")
  for (var i = 0; i < queries.length; i++) {
    var kv = queries[i].split("=");
    obj[kv[0]] = kv[1];
  }
  return obj;
}

// youtube api management

function prepareYTAPIQuery(action, params) {
  return urlUtils.format({
    protocol: 'https:',
    slashes: true,
    host: 'www.googleapis.com',
    pathname: '/youtube/v3/' + action,
    search: makeQueryString(params)
  });
}

function HTTPSAPIRequest(url, callback) {
  https.get(url, request => {
    if (request.statusCode !== 200)
      callback("Connection error", request.statusCode);
    else {
      var data = "";
      request.on('data', d => data = data + d);
      request.on('end', () => {
        var result = JSON.parse(data);
        if (result.hasOwnProperty("error")) 
          callback("api error", pagedat.error.errors);
        else {
          callback(null, result);
        }
      });
    }
  });
}

// queue manipulation

function addLinkToQueue(guild, link, callback) {
  if (!queues.hasOwnProperty(guild.id)) {
    queues[guild.id] = [];
  }
  queues[guild.id].push(link);
}

function setQueue(guild, value) {
    queues[guild.id] = value;
}

function getQueue(guild) {
  if (!queues.hasOwnProperty(guild.id))
    return [];
  return queues[guild.id];
}

// playlist information fetching

function recurGetPlaylistContents(pId, callback, vIds=[], nextPageToken=null) {
  var queryStringBase = {
    part: "snippet",
    maxresults: 50,
    playlistId: pId,
    key: tokens.youtube_api_token
  };
  if (nextPageToken) 
    queryStringBase.pageToken = nextPageToken;
  HTTPSAPIRequest(prepareYTAPIQuery("playlistItems", queryStringBase), (err, data) => {
    if (err) {
      callback(err, data);
    } else {
      var ids = [];
      for (var i = 0; i < data.items.length; i++) {
        ids.push(data.items[i].snippet.resourceId.videoId);
      }
      var newVIds = vIds.concat(ids);
      if (data.hasOwnProperty("nextPageToken")) {
        recurGetPlaylistContents(pId, callback, newVIds, data.nextPageToken);
      } else {
        callback(null, newVIds);
      }
    }
  });
}

function getPlaylistContents(playlistUrl, callback) {
  var qobj = getQueryStringObject(playlistUrl);
  if (!qobj.hasOwnProperty("list")) callback("invalid url");              
  else {
    recurGetPlaylistContents(qobj.list, (errtext, content) => {
      if (errtext) {
        log("Error fetching playlist data; " + errtext + content);
      } else {
        var vidUrls = [];
        for (var i = 0; i < content.length; i++) {
          vidUrls.push("https://www.youtube.com/watch?v=" + content[i]);
        }
        callback(vidUrls);
      }
    });
  }
}

// link type disseminating

function getLinkType(linkurl) {
  var u = urlUtils.parse(linkurl);
  if (u.hostname === "www.youtube.com") {
    if (u.pathname === "/watch")
      return "video";
    else if (u.pathname === "/playlist")
      return "playlist";
  }
  return "invalid";
}

// printing names of videos in queue

class VideoTitleFetcher {
  constructor (link, position) {
    this.link = link;
    this.name = null;
  }

  activate(callback) {
    var fetcher = this;
    var link = this.link;
    ytdl(link, (err, inf) => {
      fetcher.name = inf.title;
      callback(inf.title);
    });
  }
}

function tandemGetVideoTitles(vidlinks, maxnames=10000, callback) {
  var vidFetchers = [];
  var counter = 0;
  for (var i=0; i<Math.min(vidlinks.length, maxnames);i++) {
    vidFetchers.push(new VideoTitleFetcher(vidlinks[i], i));
    vidFetchers[i].activate(name => {
      counter++;
      if (counter === vidFetchers.length) {
        var vidnames = [];
        for (var j = 0; j < vidFetchers.length; j++) {
          vidnames.push(vidFetchers[j].name);
        }
        callback(vidnames);
      }
    });
  }
}

function printQueueNames(channel, callback=()=>{}) {
  tandemGetVideoTitles(getQueue(channel.guild), maxmessagelength, names => {
    var s = "```Current video: " + names[0] + "\nComing up:\n";
    for (var i = 1; i < names.length; i++)
      s = s + `   ${i+1}. ${names[i]}\n`;
    var qlen = getQueue(channel.guild).length
    if (qlen > maxmessagelength)
      s = s + `   ...and ${qlen - maxmessagelength} more`;
    channel.sendMessage(s + '```');
    callback();
  });
}

// playing videos in queue

function recurExhaustQueue(vchannel, tchannel, connection) {
  if (getQueue(vchannel.guild).length === 0) {
    playing = false;
    vchannel.leave();
  } else {
    playing = true;
    ytdl(getQueue(vchannel.guild)[0], (err, info) => {
      tchannel.sendMessage("Now playing: `" + info.title + "`");
      log("Started playing new song: '" + info.title + "'");
    });
    dispatch = connection.playStream(ytdl(getQueue(vchannel.guild)[0], {audioonly: true}));
    dispatch.setVolumeLogarithmic(0.3);
    dispatch.on('end', () => {
      if (playing) {
        getQueue(vchannel.guild).shift();
        recurExhaustQueue(vchannel, tchannel, connection);
      } else {
        log("Stopped playing.");
        vchannel.leave();
      }
    });
    dispatch.on('error', err => {
      log("Connection error occured; " + err);
      tchannel.sendMessage("Encountered an error while playing.");
      vchannel.leave();
      playing = false;
    });
  }
}

// main command handling event

bot.on("message", msg => {
  var commands = {

    list: (msg, args) => {
      if (args.length == 1) {
        if (getQueue(msg.guild).length == 0) {
          msg.channel.sendMessage("No videos in playlist; type `!add <video url>` to add a video.");
        } else if (getQueue(msg.guild).length == 1) {
          ytdl.getInfo(getQueue(msg.guild)[0], (err, info) => {
            msg.channel.sendMessage("Current video: `" + info.title + "`");
          });
        } else {
          msg.channel.sendMessage("Getting playlist contents...").then(mes => {
            printQueueNames(msg.channel, () => {
              mes.delete();
            });
          });
        }
      } else if (args[1] === "clear") {
        setQueue(msg.guild, []);
        msg.channel.sendMessage("Cleared queue.");
        log("Cleared queue.");
      }
    },

    add: (msg, args) => {
      if (args.length == 1) {
          msg.channel.sendMessage("Provide a YouTube video or playlist url; `!add <video url>`");
      } else {
        var linktype = getLinkType(args[1]);
        if (linktype === "invalid") {
          msg.channel.sendMessage("`" + args[1] + "` is not a valid YouTube video or playlist url.");
        } else if (linktype === "video") {
          ytdl.getInfo(args[1], (err, info) => {
            if (info) {
              getQueue(msg.guild).push(args[1]);
              msg.channel.sendMessage("Added `" + info.title + "` to queue");
              log("Added new song to queue: '" + info.title + "'");
            } else {
              log("Error fetching video information; " + err);
            }
          });
        } else if (linktype === "playlist") {
          getPlaylistContents(args[1], vids => {
            setQueue(msg.guild, getQueue(msg.guild).concat(vids));
            msg.channel.sendMessage("Added videos from playlist to queue");
            log(`Added playlist ${args[1]} to queue`);
          });
        }
      }
    },

    play: (msg, args) => {
      if (getQueue(msg.guild).length === 0) {
        msg.channel.sendMessage("No videos in queue; type `!add <video url>` to add a video.");
      } else if (!msg.member.voiceChannel) {
        msg.channel.sendMessage("Join a voice channel before using `!play`");
      } else {
        msg.member.voiceChannel.join().then( c => recurExhaustQueue(msg.member.voiceChannel, msg.channel, c));
      }
    },

    stop: (msg, args) => {  
      if (!playing) {
        msg.channel.sendMessage("Not currently playing in a channel.");
      } else {
        playing = false;
        dispatch.end();
      }
    },

    skip: (msg, args) => {
      if (getQueue(msg.guild).length === 0) {
        msg.channel.sendMessage("No videos in queue.");
      } else {
        ytdl(getQueue(msg.guild)[0], (err, inf) => {
          msg.channel.sendMessage("Skipped video `" + inf.title + "`");
          log("Skipped current song, '" + inf.title + "'");
        });
        if (playing) {
          dispatch.end();
        } else {
          getQueue(msg.guild).shift();
        }
      }
    },

    commands: (msg, args) => {
      var helptext = "```Commands:\n";
      for (var c in commands) {
        helptext = helptext + "    " + prefix + c + "\n";
      }
      msg.channel.sendMessage(helptext + '```');
    },

    help: (msg, args) => commands.commands(msg, args)
  };

  var prefix = "!";
  var text = msg.content;
  var args = text.split(" ").filter( l => l.length > 0 );
  if (args.length === 0) {
    return;
  }
  if (!args[0].startsWith(prefix)) {
    return;
  }
  var cmd = args[0].slice(1);
  if (!commands.hasOwnProperty(cmd)) {
    msg.channel.sendMessage("`" + args[0] + "` is not a valid command");
  } else {
    commands[cmd](msg, args);
  }
});

log("Connecting to discord");
bot.login(tokens.discord_api_token);