package mainpack;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.block.BlockFace;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.player.AsyncPlayerChatEvent;
import org.bukkit.plugin.java.JavaPlugin;

public class ProxyChat extends JavaPlugin implements Listener {

    public double mult;
    
    public char[] muffledCharacters = {
        '!', '@', '#', '$', '%', '^', '&', '*', '-', '+', '=', '|', '~'};

    @Override
    public void onEnable() {
        Bukkit.getPluginManager().registerEvents(this, this);

        getConfig().addDefault("whisper_range", 3.0);
        getConfig().addDefault("audible_range", 40.0);
        getConfig().addDefault("shout_range", 90.0);
        getConfig().addDefault("distance_muffle", true);
        getConfig().addDefault("falloff_multiplier", 0.5);
        getConfig().addDefault("chat_format", "<{name}> {message}");
        getConfig().addDefault("shout_format", "<{name}> {message}");
        getConfig().addDefault("whisper_format", "<{name}> {message}");
        getConfig().addDefault("muffle_format", ChatColor.DARK_GRAY + "{relative} {message}");
        getConfig().addDefault("broadcast_format", ChatColor.LIGHT_PURPLE + "[{name}] {message}");
        getConfig().addDefault("outofrange_format", "* {pos} <{name}> {message}");
        getConfig().options().copyDefaults(true);
        saveConfig();
        reloadConfig();

        mult = getConfig().getDouble("falloff_multiplier", 0.5) + 1;
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        Player ply = sender instanceof Player ? (Player) sender : null;
        if (ply == null) {
            sender.sendMessage("You must be a player to use ProxyChat commands.");
            return true;
        }

        if (args.length == 0) return false;

        String message = "";
        for (String a : args) message += a + " ";
        message = message.substring(0, message.length()-1);

        if (command.getName().equalsIgnoreCase("shout")) {
            if (ply.hasPermission("proxychat.shout")) {
                sendOut(ply, message, getConfig().getDouble("shout_range", 90.0), "shout_format");
            } else {
                sender.sendMessage("You can't shout.");
            }
        }

        else if (command.getName().equalsIgnoreCase("whisper")) {
            if (ply.hasPermission("proxychat.whisper")) {
                sendOut(ply, message, getConfig().getDouble("whisper_range", 3.0), "whisper_format");
            } else {
                sender.sendMessage("You can't whisper.");
            }
        }

        else if (command.getName().equalsIgnoreCase("broadcast")) {
            if (ply.hasPermission("proxychat.broadcast")) {
                String format = getFormat(ply, null, message, "broadcast_format");
                for (Player p : Bukkit.getOnlinePlayers()) p.sendMessage(format);
                Bukkit.getConsoleSender().sendMessage(format);
            } else {
                ply.sendMessage("You can't broadcast messages.");
            }
        }

        return true;
    }

    @EventHandler(priority = EventPriority.HIGHEST)
    public void Chat(AsyncPlayerChatEvent event) {
        event.setCancelled(true);
        sendOut(event.getPlayer(), event.getMessage(), getConfig().getDouble("audible_range", 40.0), "chat_format");
    }

    public void sendOut(Player origin, String message, double range, String formatType) {
        for (Player receiving : Bukkit.getOnlinePlayers()) {
            double distance = receiving.getLocation().distance(origin.getLocation());
            if (receiving.hasPermission("proxychat.eartotheground") && distance >= range) {
                receiving.sendMessage(getFormat(origin, receiving, message, "outofrange_format"));
            } else if (receiving.getWorld().equals(origin.getWorld()) && distance < range * mult) {
                if (distance < range) {
                    receiving.sendMessage(getFormat(origin, receiving, message, formatType));
                } else if (getConfig().getBoolean("distance_muffle", true)) {
                    double percent = ((range * mult) - distance) / (range * (mult - 1));
                    message = muffleMessage(message, percent);
                    receiving.sendMessage(getFormat(origin, receiving, message, "muffle_format"));
                }
            }
        }
        Bukkit.getConsoleSender().sendMessage(getFormat(origin, null, message, formatType));
    }

    public String getFormat(Player sender, Player receiver, String message, String formatType) {
        String toGive = getConfig().getString(formatType, "<{name}> {message}");
        toGive = toGive.replaceAll("\\{name\\}", sender.getDisplayName());
        toGive = toGive.replaceAll("\\{message\\}", message);

        Location sloc = sender.getLocation();
        String formatPos = "(" + sloc.getBlockX() + ", " + sloc.getBlockY() + ", " + sloc.getBlockZ() + ")";
        toGive = toGive.replaceAll("\\{pos\\}", formatPos);
        if (receiver != null) {
            Location direction = receiver.getLocation().subtract(sloc);
            direction.setDirection(direction.toVector().normalize());
            String dirPrint = getVirtualDirection(direction).toString();
            dirPrint = prettify(dirPrint).replaceAll(" ", "");
            toGive = toGive.replaceAll("\\{relative\\}", dirPrint);
        } else {
            toGive = toGive.replaceAll("\\{relative\\}", "N/A");
        }
        return toGive;
    }

    public BlockFace getVirtualDirection(Location direction) {
        double yaw = direction.getYaw();
        yaw = yaw > 180 ?  yaw - 360 : yaw;
        if (yaw < -157.5) return BlockFace.NORTH;
        else if (yaw < -112.5) return BlockFace.NORTH_EAST;
        else if (yaw < -67.5) return BlockFace.EAST;
        else if (yaw < -22.5) return BlockFace.SOUTH_EAST;
        else if (yaw < 22.5) return BlockFace.SOUTH;
        else if (yaw < 67.5) return BlockFace.SOUTH_WEST;
        else if (yaw < 112.5) return BlockFace.WEST;
        else if (yaw < 157.5) return BlockFace.NORTH_WEST;
        else return BlockFace.NORTH;
    }

    public String prettify(String ugly) {
        String pretty = "";
        for (String word : ugly.split("_"))
            pretty = pretty.concat(word.substring(0, 1).toUpperCase() + word.substring(1).toLowerCase() + " ");
        return pretty.substring(0, pretty.length()-1);
    }

    public String muffleMessage(String message, double percentage) {
        double scale = 0.01;
        double falloff = (Math.pow(scale, percentage) - 1) / (scale - 1);
        int charstoprint = (int) (falloff * (double) message.length());
        String shortmessage = message.substring(charstoprint/2, message.length() - charstoprint/2);
        String newmessage = "...";
        for (int i=0;i<shortmessage.length();i++) {
            if (math.random() < percentage)
                newmessage += shortmessage.charAt(i);
            else {
                newmessage += muffledCharacters[(int)(shortmessage.length() * math.random())];
            }
        }
        return newmessage + "...";
    }

}
