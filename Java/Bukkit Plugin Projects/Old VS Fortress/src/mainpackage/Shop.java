package mainpackage;

import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;


public class Shop {
	VSFortress plug;
	GameManager hostgame;
	
	Shop(VSFortress plug, GameManager hostgame) {
		this.plug = plug;
		this.hostgame = hostgame;
	}
	
	public Inventory showMain(Player p) {
		Inventory inv = plug.getServer().createInventory(null, 9, "Main Menu");
		
		ItemStack upgrades = new ItemStack(Material.GOLD_BLOCK, 1);
		upgrades.getItemMeta().setDisplayName("Upgrades");
		inv.setItem(1, upgrades);
		
		ItemStack coreups = new ItemStack(Material.GOLD_BLOCK, 1, (byte) 120);
		coreups.getItemMeta().setDisplayName("Core Upgrades");
		inv.setItem(3, coreups);
		
		ItemStack other = new ItemStack(Material.LAVA_BUCKET, 1);
		coreups.getItemMeta().setDisplayName("Other");
		inv.setItem(5, other);
		
		p.openInventory(inv);
		return inv;
	}
}
