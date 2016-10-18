const discord_api = require("discord.js");
var request = require('request');
var data = require("Bot-Info.json");

console.log("Connecting to discord..")
const bot = new discord_api.Client();
bot.login(data.discord.api_token);

console.log("Connecting to spotify..")
var spotify_api_token = ""
request.post({
  url: 'https://accounts.spotify.com/api/token',
  headers: { 'Authorization': 'Basic ' + (new Buffer(data.spotify.id_token + ':' + data.spotify.secret_token).toString('base64')) },
  form: { grant_type: 'client_credentials' },
  json: true
}, function(err, response, body) {
  if (!error && response.satusCode === 200) {
    spotify_api_token = body.access_token;
  }
});

console.log("Successfully connected to all servcies")

bot.on("message", evargs => {
  var text = evargs.content;
  if (text.indexOf("!radio")) {
    
  }
});

request.get({
  url: 'https://api.spotify.com/v1/',
  headers: { 'Authorization': 'Bearer ' + token },
  json: true
}, function(err, resp, body) {
  console.log();
});