package mainpackage;

import java.util.ArrayList;

import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;

public class MainFile extends JavaPlugin{
	
	public static String badPermissionsMessage = ChatColor.RED + "You don't have permission to use that command.";
	public static String badInterfaceMessage = ChatColor.RED + "You must be an active player to use that command.";
	public static String maindir = "plugins/Basic_Scores";
	public static String[] values = {
		"score"
	};
	public static PlayerInfo plyInfo = new PlayerInfo(values, maindir);
	public static Ticker ticker = new Ticker();
	
	public void onEnable() {
		getServer().getPluginManager().registerEvents(new listen(), this);
		getServer().getScheduler().scheduleSyncRepeatingTask(this, new Runnable() {public void run() {
			ticker.run();	
		}}, 0, 1);
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
		Player player = null;
		if (!isCmd) { player = (Player) sender; }
		
		// tps
		if (command.equalsIgnoreCase("tps")) {
			if (!hasPow) {
				sender.sendMessage(badPermissionsMessage);
			} else {
				sender.sendMessage("Ticks Per Second Registered: " + ticker.tps());
			}
		}
		
		// givedirt
		else if (command.equalsIgnoreCase("givedirt")) {
			if (isCmd) {
				sender.sendMessage(badInterfaceMessage);
			} else if (!isOp) {
				sender.sendMessage(badPermissionsMessage);
			} else {
				player.getInventory().addItem(new ItemStack(Material.DIRT, 64));
			}
		}
		
		// checkscore
		else if (command.equalsIgnoreCase("checkscore")) {
			if (isCmd) {
				sender.sendMessage(badInterfaceMessage);
			} else {
				sender.sendMessage("Your score is " + plyInfo.getValue(player, "score") + ".");
			}
		}
		
		// save-data
		else if (command.equalsIgnoreCase("save-data")) {
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
		
		// blockonhead
		else if (command.equalsIgnoreCase("blockonhead")) {
			if (isCmd) {
				sender.sendMessage(badInterfaceMessage);
			} else {
				Location loc = player.getLocation();
				loc.add(0, 2, 0);
				Block b = player.getWorld().getBlockAt(loc);
				if (b.getType() == Material.AIR) {
					b.setType(Material.DIRT);
				} else {
					sender.sendMessage("There's already a block on your head!");
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
	public void onBlockBreak(BlockBreakEvent event) {
		Player p = event.getPlayer();
		int oldamount = MainFile.plyInfo.getValue(p, "score");
		MainFile.plyInfo.setValue(p, "score", oldamount+1);
	}
	
	@EventHandler
	public void onBlockPlace(BlockPlaceEvent event) {
		
	}
	
	@EventHandler
	public void playerInteract(PlayerInteractEvent event) {
		
	}
	
	@EventHandler
	public void playerJoin(PlayerJoinEvent event) {
		MainFile.plyInfo.register(event.getPlayer());
		event.getPlayer().sendMessage("Hello, " + ChatColor.GOLD + event.getPlayer().getName() + ChatColor.RESET + "!");
	}
	
	@EventHandler
	public void playerLeave(PlayerQuitEvent event) {
		MainFile.plyInfo.save(event.getPlayer());
		MainFile.plyInfo.info.remove(event.getPlayer());
	}
}

class Ticker {
	ArrayList<Long> ticks = new ArrayList<Long>();
	
	public int tps() {
		return this.ticks.size();
	}
	
	public void run() {
		this.ticks.add(System.currentTimeMillis());
		int i = 0;
		while (i < this.ticks.size()) {
			if (this.ticks.get(i) < System.currentTimeMillis()-1000) {
				this.ticks.remove(i);
				i--;
			}
			i++;
		}
	}
}