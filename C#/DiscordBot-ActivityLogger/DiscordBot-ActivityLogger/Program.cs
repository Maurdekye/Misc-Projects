using Discord;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DiscordBot_ActivityLogger
{
    class Program
    {
        static void Main(string[] args)
        {
            new ActivityLogger();
        }
    }

    class ActivityLogger
    {
        public async void LogString(string text)
        {
            text = $"[{DateTime.UtcNow.ToLongTimeString()}] {text}";
            Console.WriteLine(text);
            if (!Directory.Exists("logs"))
                Directory.CreateDirectory("logs");
            string logFileName = "logs/" + DateTime.UtcNow.ToShortDateString().Replace("/", "-") + ".log";
            if (!File.Exists(logFileName))
                File.Create(logFileName).Close();
            StreamWriter writer = File.AppendText(logFileName);
            await writer.WriteLineAsync(text);
            writer.Close();
        }

        public ActivityLogger()
        {
            DiscordClient Client = new DiscordClient(Logger =>
            {
                Logger.LogLevel = LogSeverity.Info;
                Logger.LogHandler += delegate (object Sender, LogMessageEventArgs EvArgs)
                {
                    Console.WriteLine(EvArgs.Message);
                };
            });

            Client.UserUpdated += (object sender, UserUpdatedEventArgs EvArgs) =>
            {
                if (EvArgs.Before.VoiceChannel != EvArgs.After.VoiceChannel)
                {
                    if (EvArgs.Before.VoiceChannel == null)
                    {
                        LogString($"User {EvArgs.Before} joined voice channel {EvArgs.After.VoiceChannel}");
                    }
                    else if (EvArgs.After.VoiceChannel == null)
                    {
                        LogString($"User {EvArgs.Before} left voice channel {EvArgs.Before.VoiceChannel}");
                    }
                    else
                    {
                        LogString($"User {EvArgs.Before} changed from voice channel {EvArgs.Before.VoiceChannel} to {EvArgs.After.VoiceChannel}");
                    }
                }
                if (EvArgs.Before.Nickname != EvArgs.After.Nickname)
                {
                    if (EvArgs.Before.Nickname == null)
                    {
                        LogString($"User {EvArgs.Before} gave themselves the nickname {EvArgs.After.Nickname}");
                    }
                    else if (EvArgs.After.Nickname == null)
                    {
                        LogString($"User {EvArgs.Before} removed their nickname {EvArgs.Before.Nickname}");
                    }
                    else
                    {
                        LogString($"User {EvArgs.Before} changed their nickname from {EvArgs.Before.Nickname} to {EvArgs.After.Nickname}");
                    }
                }
            };

            Console.WriteLine("Connecting to discord");

            Client.ExecuteAndWait(async () =>
            {
                while (true)
                {
                    try
                    {
                        await Client.Connect(ActivityLoggerTokenCarrier.Token, TokenType.Bot);
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
}
