import org.bukkit.ChatColor;
import org.bukkit.Material;
import org.bukkit.Sound;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.AsyncPlayerChatEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitScheduler;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Created by ncola on 4/4/2016.
 */
public class Main extends JavaPlugin implements Listener, CommandExecutor {

    private List<String> words;
    private final int normalgametime = 10;
    private final int intermissiontime = 5;
    private final int scrambledgametime = 20;

    int globalScheduleId = 0;
    private String currentword;
    private boolean running = false;
    private boolean playing = false;
    private GameType gametype;
    private BukkitScheduler calendar;

    public void onEnable()
    {
        getConfig().addDefault("words", Arrays.asList("jaunt", "haggard", "jellybean", "suture", "barnacle", "calgary", "tencent", "parsnip", "caliber", "ludicrous", "sanctuary"));
        getConfig().options().copyDefaults(true);
        saveConfig();
        words = getConfig().getStringList("words");

        calendar = getServer().getScheduler();
        getServer().getPluginManager().registerEvents(this, this);

    }

    public boolean onCommand(CommandSender sender, Command command, String label, String[] args)
    {
        if (command.getName().equals("startgame"))
        {
            if (running)
            {
                sender.sendMessage("Game is already running.");
                return true;
            }
            newGame();
        }
        else if (command.getName().equals("stopgame"))
        {
            if (!running)
            {
                sender.sendMessage("No game currently running.");
                return true;
            }
            endGame();
        }
        return true;
    }

    public void endGame()
    {
        playing = false;
        running = false;
        getServer().broadcastMessage(ChatColor.BLUE + "The game has been stopped.");
        calendar.cancelTask(globalScheduleId);
    }

    public void finishNoAnswer()
    {
        playing = false;
        if (gametype == GameType.NORMAL)
            getServer().broadcastMessage(ChatColor.DARK_AQUA + "Nobody typed the word in time. New game will start in " + intermissiontime + " seconds.");
        else if (gametype == GameType.SCRAMBLED)
            getServer().broadcastMessage(ChatColor.DARK_AQUA + "Nobody guessed the word; it was '" + ChatColor.AQUA + currentword + ChatColor.DARK_AQUA + "'. New game will start in " + intermissiontime + " seconds.");
        globalScheduleId = calendar.scheduleSyncDelayedTask(this, new Runnable() {
            @Override
            public void run() {
                newGame();
            }
        }, 20*intermissiontime);
    }

    public void finishCorrectGuess(Player guesser)
    {
        playing = false;
        guesser.getInventory().addItem(new ItemStack(Material.GOLD_BLOCK, 1));
        guesser.playSound(guesser.getLocation(), Sound.ENTITY_PLAYER_LEVELUP, 1, 1);
        if (gametype == GameType.SCRAMBLED)
            getServer().broadcastMessage(ChatColor.YELLOW + guesser.getName() + ChatColor.DARK_AQUA + " guessed the word; it was '" + ChatColor.AQUA + currentword + ChatColor.DARK_AQUA + "'. A new game will begin in " + intermissiontime + " seconds.");
        else if (gametype == GameType.NORMAL)
            getServer().broadcastMessage(ChatColor.YELLOW + guesser.getName() + ChatColor.DARK_AQUA + " typed the word. A new game will begin in " + intermissiontime + " seconds.");
        calendar.cancelTask(globalScheduleId);
        globalScheduleId = calendar.scheduleSyncDelayedTask(this, new Runnable() {
            @Override
            public void run() {
                newGame();
            }
        }, 20*intermissiontime);
    }

    public void newNormalGame()
    {
        getServer().broadcastMessage(ChatColor.GOLD + "Type the word: '" + ChatColor.AQUA + currentword + ChatColor.GOLD + "'. You have " + normalgametime + " seconds.");
        gametype = GameType.NORMAL;
    }

    public void newScrambledGame()
    {
        getServer().broadcastMessage(ChatColor.GOLD + "Guess what the scrambled word is: '" + ChatColor.AQUA + scramble(currentword) + ChatColor.GOLD + "'. You have " + scrambledgametime + " seconds.");
        gametype = GameType.SCRAMBLED;
    }

    public void newGame()
    {
        playing = true;
        running = true;
        currentword = words.get((int) (Math.random() * words.size()));
        if (Math.random() > 0.5)
            newScrambledGame();
        else
            newNormalGame();
        int gametime = 0;
        if (gametype == GameType.SCRAMBLED)
            gametime = scrambledgametime;
        if (gametype == GameType.NORMAL)
            gametime = normalgametime;
        globalScheduleId = calendar.scheduleSyncDelayedTask(this, new Runnable() {
            @Override
            public void run() {
                finishNoAnswer();
            }
        }, 20*gametime);
    }

    @EventHandler
    public void AsyncPlayerChat(AsyncPlayerChatEvent e)
    {
        if (playing)
        {
            if (e.getMessage().trim().equalsIgnoreCase(currentword))
            {
                finishCorrectGuess(e.getPlayer());
                e.setCancelled(true);
            }
        }
    }

    public String scramble(String s)
    {
        char[] letters = s.toCharArray();
        for (int i=0;i<s.length();i++)
        {
            for (int j=0;j<s.length();j++)
            {
                int swapIndex = (int) (Math.random() * letters.length);
                char placeholder = letters[j];
                letters[j] = letters[swapIndex];
                letters[swapIndex] = placeholder;
            }
        }
        String retStr = "";
        for (char c : letters)
            retStr += c;
        return retStr;
    }

    public int weightedRandom(double[] numbers)
    {
        double sum = 0;
        for (double f : numbers)
            sum += f;
        double rand = Math.random() * sum;
        for (int i=0;i<numbers.length;i++)
        {
            if (rand < numbers[i])
                return i;
            rand -= numbers[i];
        }
        return numbers.length - 1;
    }
}

enum GameType
{
    NORMAL, SCRAMBLED
}

