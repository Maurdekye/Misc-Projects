using Discord;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace DiscordBot_OfficeBot
{
    class Program
    {
        static void Main(string[] args)
        {
            new OfficeBot();
        }
    }
}

class OfficeBot
{

    public Dictionary<Channel, List<User>> GetOfficeChannels(Server NewServer)
    {
        //Console.WriteLine("Scanning for office channels");

        Dictionary<Channel, List<User>> OfficeChannels = new Dictionary<Channel, List<User>>();
        foreach (Channel ServerChannel in NewServer.AllChannels)
        {
            if (ServerChannel.Type == ChannelType.Voice)
            {
                if (ServerChannel.Name.Contains("Office"))
                {
                    Match NameMatch = Regex.Match(ServerChannel.Name, "(.+?)'s Office");
                    if (NameMatch.Success)
                    {
                        List<User> OfficeUsers = NewServer.FindUsers(NameMatch.Groups[1].Value, exactMatch: true).Where(U => U.ServerPermissions.MoveMembers).ToList();
                        if (OfficeUsers.Count > 0)
                        {
                            OfficeChannels[ServerChannel] = OfficeUsers;
                            //await ServerChannel.Edit(maxusers: 2);
                            //Console.WriteLine(($"Registered Office Channel '{ServerChannel}' with users [{string.Join(", ", OfficeUsers)}]"));
                        }
                    }
                }
            }
        }
        return OfficeChannels;
    }

    public OfficeBot()
    {

        // Startup Code

        DiscordClient Client = new DiscordClient(Logger =>
        {
            Logger.LogLevel = LogSeverity.Info;
            Logger.LogHandler += delegate (object Sender, LogMessageEventArgs EvArgs)
            {
                Console.WriteLine(EvArgs.Message);
            };
        });

        // Check and prevent user from moving to office channel if he does not own it

        Client.UserUpdated += async (object Sender, UserUpdatedEventArgs EvArgs) =>
        {
            if (EvArgs.After.VoiceChannel.Users.Count() > 1 && EvArgs.Before.VoiceChannel != EvArgs.After.VoiceChannel)
            {
                Dictionary<Channel, List<User>> OfficeChannels = GetOfficeChannels(EvArgs.Server);
                if (OfficeChannels.ContainsKey(EvArgs.After.VoiceChannel))
                {
                    List<User> ValidUsers = OfficeChannels[EvArgs.After.VoiceChannel];
                    if (!ValidUsers.Contains(EvArgs.After))
                    {
                        await EvArgs.Before.Edit(voiceChannel: EvArgs.Before.VoiceChannel);
                    }
                }
            }
        };

        Client.ExecuteAndWait(async () =>
        {
            while (true)
            {
                try
                {
                    await Client.Connect(DiscordBot_OfficeBot.OfficeBotTokenCarrier.Token, TokenType.Bot);
                    break;
                }
                catch
                {
                    await Task.Delay(3000);
                }
            }
        });
    }
}
