const discord = require("discord.js");
const ytdl = require('ytdl-core');
const fs = require('fs');
const https = require('https');
const urlUtils = require('url');

// global vars

const maxMessageLength = 2000;
const queueSaveFile = "vidqueues.json";

var tokens = JSON.parse(fs.readFileSync("botinfo.json"));
var queues = new discord.Collection();
var playings = new discord.Collection();
var dispatches = new discord.Collection();

const bot = new discord.Client();

// read songs from file

if (fs.access(queueSaveFile, fs.constants.R_OK, err => {
  if (!err) {
    var rawlistdata = JSON.parse(fs.readFileSync(queueSaveFile));
    for (var guildid in rawlistdata) {
      queues.set(guildid, rawlistdata[guildid]);
    }
  }
}));

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

// general util functions

function formatTimeString(seconds) {
  if (seconds < 60)
    return seconds + "s";
  var minutes = Math.floor(seconds / 60);
  seconds = seconds % 60;
  var hours = 0;
  if (minutes >= 60) {
    hours = Math.floor(minutes / 60);
    minutes = minutes % 60;
  }
  if (seconds < 10) seconds = '0' + seconds;
  if (minutes < 10 && hours > 0) minutes = '0' + minutes;
  var fstring = `${minutes}:${seconds}`;
  if (hours > 0)
    fstring = `${hours}:${fstring}`;
  return fstring;
}

function replaceAll(string, find, rep) {
  while (string.indexOf(find) >= 0)
    string = string.replace(find, rep);
  return string;
};

function repMulti(string, find, rep) {
  for (var i = 0; i < find.length; i++) {
    string = replaceAll(string, find[i], rep[i]);
  }
  return string;
}

function clean(string, invalid) {
  for (var i = 0; i < invalid.length;i++) {
    string = replaceAll(string, invalid[i], "");
  }
  return string;
}

function collectionToObj(col) {
  var obj = {};
  for (const kv of col) {
    obj[kv[0]] = kv[1];
  }
  return obj;
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
  log("HTTP Request to " + url);
  https.get(url, request => {
    if (request.statusCode !== 200)
      callback("Connection error", request.statusCode);
    else {
      var data = "";
      request.on('data', d => data = data + d);
      request.on('end', () => {
        var result = JSON.parse(data);
        if (result.hasOwnProperty("error")) 
          callback("API error", pagedat.error.message);
        else {
          callback(null, result);
        }
      });
    }
  });
}

function YTAPIQR(action, params, callback) {
  HTTPSAPIRequest(prepareYTAPIQuery(action, params), callback);
}

function videoName(url, callback) {
  ytdl.getInfo(url, (err, info) => {
    if (err)
      callback(err, info);
    else {
      callback(null, `${info.title} (${formatTimeString(info.length_seconds)})`);
    }
  })
}

// multiserver information manipulation

function saveQueues() {
  var jsonstring = JSON.stringify(collectionToObj(queues))
  fs.writeFileSync(queueSaveFile, jsonstring);
}

function addLinkToQueue(guild, link, callback) {
  if (!queues.has(guild.id)) {
    queues.set(guild.id, []);
  }
  setQueue(guild, queues.get(guild.id).concat([link]));
}

function setQueue(guild, value) {
  queues.set(guild.id, value);
  saveQueues();
}

function getQueue(guild) {
  if (!queues.has(guild.id))
    return [];
  return queues.get(guild.id);
}

function playing(guild) {
  if (!playings.has(guild.id))
    return false;
  return playings.get(guild.id);
}

function setPlaying(guild, value) {
  playings.set(guild.id, value);
}

// youtube api interactions

function recurGetPlaylistContents(pId, callback, vIds=[], nextPageToken=null) {
  var queryStringBase = {
    part: "snippet",
    maxresults: 50,
    playlistId: pId,
    key: tokens.youtube_api_token
  };
  if (nextPageToken) 
    queryStringBase.pageToken = nextPageToken;
  YTAPIQR("playlistItems", queryStringBase, (err, data) => {
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

function getPlaylistTitle(playlistUrl, callback) {
  var qobj = getQueryStringObject(playlistUrl);
  if (!qobj.hasOwnProperty("list")) callback("invalid url");              
  else {
    var queryStringBase = {
      part: 'snippet',
      id: qobj.list,
      key: tokens.youtube_api_token
    }
    YTAPIQR("playlists", queryStringBase, (err, info) => {
      if (err)
        console.log("Error getting playlist title: " + err + info);
      else {
        callback(info.items[0].snippet.title);
      }
    });
  }
}

function searchYoutubeVideo(searchterms, callback) {
  var cleansearch = replaceAll(clean(searchterms, "\n\t-+=\\/&?\"'"), " ", "+");
  var queryStringBase = {
    part: 'snippet',
    q: cleansearch,
    maxresults: 20,
    key: tokens.youtube_api_token
  }
  YTAPIQR("search", queryStringBase, (err, info) => {
    if (err)
      console.log("Error searching: " + err + info);
    else {
      if (info.pageInfo.totalResults === 0) {
        callback(null);
      } else {
        var found = false;
        for (var i = 0; i < 20; i++) {
          if (info.items[i].id.kind === "youtube#video") {
            callback("https://www.youtube.com/watch?v=" + info.items[i].id.videoId);
            found = true;
            break;
          }
        }
        if (!found)
          callback(null);
      }
    }
  });
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
    videoName(link, (err, title) => {
      fetcher.name = title;
      callback(title);
    });
  }
}

function tandemGetVideoTitles(vidlinks, callback) {
  var vidFetchers = [];
  var counter = 0;
  for (var i=0; i<vidlinks.length;i++) {
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
  tandemGetVideoTitles(getQueue(channel.guild), names => {
    var s = "```Current video: " + names[0] + "\nComing up:\n";
    for (var i = 1; i < names.length; i++) {
      s = s + `   ${i+1}. ${names[i]}\n`;
      if (s.length > maxMessageLength)
        break;
    }
    if (s.length > maxMessageLength) {
      var qlen = getQueue(channel.guild).length;
      var endbit = `   ...and ${qlen - i} more`;
      while ((s + endbit + '```').length > maxMessageLength) {
        s = s.substring(0, s.lastIndexOf("\n"));
        i--;
        endbit = `\n   ...and ${qlen - i} more`;
      }
      s = s + endbit;
    }
    channel.sendMessage(s + '```');
    callback();
  });
}

// playing videos in queue

function recurExhaustQueue(vchannel, tchannel, connection) {
  if (getQueue(vchannel.guild).length === 0) {
    setPlaying(vchannel.guild, false);
    vchannel.leave();
  } else {
    setPlaying(vchannel.guild, true);
    videoName(getQueue(vchannel.guild)[0], (err, title) => {
      tchannel.sendMessage("Now playing: `" + title + "`");
      log("Started playing new song: '" + title + "'");
    });
    dispatches.set(vchannel.guild.id, connection.playStream(ytdl(getQueue(vchannel.guild)[0], {audioonly: true})));
    dispatches.get(vchannel.guild.id).setVolumeLogarithmic(0.5);
    dispatches.get(vchannel.guild.id).on('end', () => {
      if (playing(vchannel.guild)) {
        getQueue(vchannel.guild).shift();
        recurExhaustQueue(vchannel, tchannel, connection);
      } else {
        log("Stopped playing.");
        vchannel.leave();
      }
    });
  }
}

// main command handling event

function addVideo(msg, args, callback) {
  if (args.length == 1) {
    msg.channel.sendMessage("Provide a YouTube video or playlist url; `!add <video url>`");
  } else {
    var linktype = getLinkType(args[1]);
    if (linktype === "invalid") {
      searchYoutubeVideo(args.slice(1).join(" "), vidlink => {
        if (vidlink === null) {
          msg.channel.sendMessage("Could not find video.");
        } else {
          videoName(vidlink, (err, title) => {
            if (!err) {
              addLinkToQueue(msg.guild, vidlink);
              msg.channel.sendMessage("Added `" + title + "` to queue");
              log("Added new video to queue: '" + title + "'");
              callback();
            } else {
              log("Error fetching video information; " + err);
            }
          });
        }
      });
    } else if (linktype === "video") {
      videoName(args[1], (err, title) => {
        if (!err) {
          addLinkToQueue(msg.guild, args[1]);
          msg.channel.sendMessage("Added `" + title + "` to queue");
          log("Added new video to queue: '" + title + "'");
          callback();
        } else {
          log("Error fetching video information; " + err);
        }
      });
    } else if (linktype === "playlist") {
      getPlaylistContents(args[1], vids => {
        getPlaylistTitle(args[1], name => {
          setQueue(msg.guild, getQueue(msg.guild).concat(vids));
          msg.channel.sendMessage("Added playlist `" + name + "` to queue");
          log(`Added playlist ${name} to queue`);
          callback();
        });
      });
    }
  }
}

bot.on("message", msg => {
  var commands = {

    list: (msg, args) => {
      if (args.length == 1) {
        if (getQueue(msg.guild).length == 0) {
          msg.channel.sendMessage("No videos in playlist; type `!add <video url>` to add a video.");
        } else if (getQueue(msg.guild).length == 1) {
          videoName(getQueue(msg.guild)[0], (err, title) => {
            msg.channel.sendMessage("Current video: `" + title + "`");
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

    add: (msg, args) => addVideo(msg, args, () => {}),

    play: (msg, args) => {
      if (playing(msg.guild)) {
        setPlaying(msg.guild, false);
        dispatches.get(msg.guild.id).end();
      }
      if (!msg.member.voiceChannel) {
        msg.channel.sendMessage("Join a voice channel before using `!play`");
      } else {
        var playVid = () => {
          if (getQueue(msg.guild).length === 0) {
            msg.channel.sendMessage("No videos in queue; type `!add <video url>` to add a video.");
          } else {
            msg.member.voiceChannel.join().then( c => recurExhaustQueue(msg.member.voiceChannel, msg.channel, c));
          }
        }
        if (args.length > 1 && getQueue(msg.guild).length == 0) {
          addVideo(msg, args, playVid);
        } else {
          playVid();
        }
      }
    },

    stop: (msg, args) => {  
      if (!playing(msg.guild)) {
        msg.channel.sendMessage("Not currently playing in a channel.");
      } else {
        setPlaying(msg.guild, false);
        dispatches.get(msg.guild.id).end();
      }
    },

    skip: (msg, args) => {
      if (getQueue(msg.guild).length === 0) {
        msg.channel.sendMessage("No videos in queue.");
      } else {
        videoName(getQueue(msg.guild)[0], (err, title) => {
          msg.channel.sendMessage("Skipped video `" + title + "`");
          log("Skipped current song, '" + title + "'");
        });
        if (playing(msg.guild)) {
          dispatches.get(msg.guild).end();
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

// init

log("Connecting to discord");
bot.login(tokens.discord_api_token);