package mainpackage;

import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.Inventory;

public class Listen implements Listener {

	VSFortress plug;
	GameManager hostgame;
	Shop shop;

	Listen(VSFortress plug, GameManager hostgame, Shop shop) {
		this.plug = plug;
		this.hostgame = hostgame;
		this.shop = shop;
	}

	@EventHandler
	public void Join(PlayerJoinEvent event) {
		if (hostgame.board.getPlayerTeam(event.getPlayer()) == null) {
			Location spawn = plug.getLocate("lobby_spawn");
			event.getPlayer().teleport(spawn);
		} else {
			hostgame.addPlayer(event.getPlayer());
		}
	}

	@EventHandler
	public void Leave(PlayerQuitEvent event) {
		DoubleTeam left = hostgame.decommit(event.getPlayer());
		if (left != null) {
			for (String name : left.members) {
				Player ply = plug.getServer().getPlayer(name);
				ply.sendMessage(event.getPlayer().getName()
						+ " has left your team!");
			}
		}
	}
	
	@EventHandler
	public void InvClick(InventoryClickEvent event) {
		if (!(event.getWhoClicked() instanceof Player)) return;
		Player ply = (Player) event.getWhoClicked();
		Inventory clicked = event.getInventory();
		if (clicked.getName().equals("Main Menu")) {
			event.setCancelled(true);
			if (event.getCurrentItem().getItemMeta().getDisplayName().equals("Upgrades")) {
				ply.sendMessage("Chosen upgrades menu.");
				ply.closeInventory();
			}
		}
	}

	@EventHandler
	public void Click(PlayerInteractEvent event) {
		if (event.getItem() == null) return;
		if (event.getItem().getType() == Material.SLIME_BALL) {
			shop.showMain(event.getPlayer());
		}
	}
}
