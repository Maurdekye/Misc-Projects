package mainpackage;

import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scoreboard.Team;
import org.bukkit.util.Vector;

public class VSFortress extends JavaPlugin {

	public GameManager game = new GameManager(this);
	public Shop shop = new Shop(this, game);
	public World hostWorld;

	public void onEnable() {
		getServer().getPluginManager().registerEvents(new Listen(this, game, shop),
				this);

		hostWorld = getServer()
				.getWorld(getConfig().getString("main_world"));
		getConfig().addDefault("lobby_spawn",
				new Vector(500, hostWorld.getHighestBlockYAt(500, 500), 500));
		int i = 0;
		for (ChatColor colour : game.usableColours) {
			getConfig()
					.addDefault(
							colour.name().toLowerCase() + "_team_spawn",
							new Vector(0, hostWorld.getHighestBlockYAt(0, i * 50),
									i * 50));
			getConfig()
			.addDefault(
					colour.name().toLowerCase() + "_team_core",
					new Vector(10, hostWorld.getHighestBlockYAt(0, i * 50),
							i * 50));
			i++;
		}

		getConfig().options().copyDefaults(true);
		saveConfig();

		for (Team t : getServer().getScoreboardManager().getMainScoreboard()
				.getTeams())
			t.unregister();
	}

	public Location getLocate(String path) {
		Vector spawnVec = getConfig().getVector(path).add(new Vector(0.5, 0.5, 0.5));
		if (spawnVec == null)
			getServer().getConsoleSender().sendMessage(ChatColor.RED + "No entry under path '" + path + "'!" );
		if (hostWorld == null)
			getServer().broadcastMessage(ChatColor.GOLD + "All is null and none is right boo hoo");
		return new Location(hostWorld, spawnVec.getX(),
				spawnVec.getY(), spawnVec.getZ());
	}

	public void onDisable() {
		game.destroy();
	}

	public boolean onCommand(CommandSender sender, Command cmd, String command,
			String[] args) {
		Player ply = (Player) sender;
		
		if (command.equalsIgnoreCase("join")) {
			game.addPlayer(ply);
		}
		
		else if (command.equalsIgnoreCase("leave")) {
			game.removePlayer(ply);
		}
		
		return true;
	}
}
