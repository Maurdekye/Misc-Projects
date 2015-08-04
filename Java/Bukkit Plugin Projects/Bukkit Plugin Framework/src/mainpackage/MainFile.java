package mainpackage;

import org.bukkit.plugin.java.JavaPlugin;

package mainpackage;
import org.bukkit.plugin.java.JavaPlugin;


import org.bukkit.ChatColor;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.plugin.java.JavaPlugin;

public class MainFile extends JavaPlugin {
	
	public static String badPermissionsMessage = ChatColor.RED + "You don't have permission to use that command.";
	public static String badInterfaceMessage = ChatColor.RED + "You must be an active player to use that command.";
	public static String maindir = "plugins/default_folder";
	public static String[] values = {};
	public static PlayerInfo plyInfo = new PlayerInfo(values, maindir);
	
	public void onEnable() {
		getServer().getPluginManager().registerEvents(new listen(), this);
		for (World w : getServer().getWorlds()) {
			for (Player p : w.getPlayers()) {
				plyInfo.register(p);
			}
		}
	}
	
	public void onDisable() {
		if (plyInfo.save_all()) {
			getLogger().info("Player information has been saved.");
		} else {
			getLogger().info("Player Information could not be saved properly.");
		}
	}
	
	public static boolean hasInt(int[] list, int lookfor) {
		for (int s : list) {if (s == lookfor) {return true;}}
		return false;
	}
	
	public boolean onCommand(CommandSender sender, Command cmd, String command, String[] args) {
		boolean isCmd = !(sender instanceof Player);
		boolean isOp = !isCmd && ((Player) sender).isOp();
		boolean hasPow = isCmd || isOp;
		
		// save-data
		if (command.equalsIgnoreCase("save-data")) {
			if (!hasPow) {
				sender.sendMessage(badPermissionsMessage);
			} else {
				if (plyInfo.save_all()) {
					sender.sendMessage(ChatColor.GREEN + "Player information successfully saved to "+maindir+".");
				} else {
					sender.sendMessage(ChatColor.RED + "Player information could not be successfully saved.");
				}
			}
		}
		
		// getinfo
		else if (command.equalsIgnoreCase("getinfo")) {
			if (!hasPow) {
				sender.sendMessage(badPermissionsMessage);
			} else {
				sender.sendMessage(ChatColor.GOLD + "Info on Server " + this.getServer().getName());
				String wnames = "";
				for (World w : this.getServer().getWorlds()) {wnames += w.getName() + ", ";}
				sender.sendMessage("Worlds: " + wnames.substring(0, wnames.length()-2));
				sender.sendMessage("Server IP: " + this.getServer().getIp());
				int p = 0;
				for (World w : this.getServer().getWorlds()) {p += w.getPlayers().size();}
				sender.sendMessage("Player Count: " + p + "/" + this.getServer().getMaxPlayers());
				sender.sendMessage("Player Information Directory: " + plyInfo.save_folder.getPath());
			}
		}
		
		else {
			sender.sendMessage("Command \"/" + command + "\" has not been implemented yet.");
		}
		return true;
	}
	
}

class listen implements Listener {
	
	@EventHandler
	public void playerJoin(PlayerJoinEvent event) {
		MainFile.plyInfo.register(event.getPlayer());
	}
	
	@EventHandler
	public void playerLeave(PlayerQuitEvent event) {
		MainFile.plyInfo.save(event.getPlayer());
		MainFile.plyInfo.info.remove(event.getPlayer());
	}
}