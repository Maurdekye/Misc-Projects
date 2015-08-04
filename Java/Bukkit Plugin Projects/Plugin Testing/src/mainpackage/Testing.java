package mainpackage;

import java.util.Arrays;
import java.util.HashMap;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.Vector;

public class Testing extends JavaPlugin implements Listener {
	
	public static HashMap<String, Location> jump_locate = new HashMap<String, Location>();
	public static HashMap<String, Boolean> jump_onground = new HashMap<String, Boolean>();
	
	public void onEnable() {
		getServer().getPluginManager().registerEvents(this, this);
		
		for (Player ply : Bukkit.getOnlinePlayers()) {
			jump_locate.put(ply.getName(), ply.getLocation());
			jump_onground.put(ply.getName(), ((Entity) ply).isOnGround());
		}
	}
	
	public void onDisable() {}
	
	public String capitalize(String instr) {
		return instr.substring(0, 1).toUpperCase() + instr.substring(1).toLowerCase();
	}
	
	public boolean onCommand(CommandSender sender, Command cmd, String cname, String[] args) {
		
		if (cname.equalsIgnoreCase("battlearmor")) {
			if (!(sender instanceof Player)) {
				sender.sendMessage("The console can't use this command!");
			}
			if (args.length == 0) {
				sender.sendMessage("You need to provide an argument; 'iron', 'gold', or 'diamond'");
				return false;
			}
			if (!Arrays.asList("gold", "iron", "diamond").contains(args[0])) {
				sender.sendMessage("Your argument must be one of these; 'iron', 'gold', or 'diamond'");
				return false;
			}
			
			String pretext = args[0].toUpperCase();
			ItemStack[] contents = new ItemStack[4];
			int i = 0;
			for (String peice : Arrays.asList("HELMET", "CHESTPLATE", "LEGGINGS", "BOOTS")) {
				ItemStack armor = new ItemStack(Material.getMaterial(pretext + "_" + peice));
				ItemMeta meta = armor.getItemMeta();
				meta.setDisplayName("Battle Ready " + capitalize(peice) + "!");
				armor.setItemMeta(meta);
				armor.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 4);
				contents[i] = armor;
				i++;
			}
			ItemStack sword = new ItemStack(Material.getMaterial(pretext + "_SWORD"));
			ItemMeta meta = sword.getItemMeta();
			meta.setDisplayName("Battle Ready Sword!");
			sword.setItemMeta(meta);
			sword.addEnchantment(Enchantment.DAMAGE_ALL, 4);
			((Player) sender).getInventory().addItem(sword);
			((Player) sender).getInventory().setArmorContents(contents);
			sender.sendMessage(ChatColor.GREEN + "Enjoy your " + args[0].toLowerCase() + " gear!");
		}
		
		return false;
	}
	
	@EventHandler
	public void Join(PlayerJoinEvent event) {
		Player ply = event.getPlayer();
		jump_locate.put(ply.getName(), ply.getLocation());
		jump_onground.put(ply.getName(), ((Entity) ply).isOnGround());
	}
	
	@EventHandler
	public void Move(PlayerMoveEvent event) {
		Player ply = event.getPlayer();
		while (((Entity) ply).isOnGround() != jump_onground.get(ply.getName())) {
			if (ply.getLocation().getY() <= jump_locate.get(ply.getName()).getY()) break;
			if (ply.isSneaking()) break;
			if (!ply.isOp()) break;
			
			ply.setVelocity(ply.getVelocity().add(new Vector(0, 0.6, 0).add(ply.getLocation().getDirection())));
			
			break;
			
		}
		jump_locate.put(ply.getName(), ply.getLocation());
		jump_onground.put(ply.getName(), ((Entity) ply).isOnGround());
	}
	
	@EventHandler
	public void Death(PlayerDeathEvent event) {
		for (ItemStack item : event.getDrops()) {
			event.getEntity().getWorld().dropItem(event.getEntity().getLocation(), item);
		}
		
	}
	
	@EventHandler
	public void Click(PlayerInteractEvent event) {
		if (event.getItem() == null) return;
		if (event.getMaterial() == Material.WHEAT) {
			Inventory show = getServer().createInventory(null, 45, "Blah blah buuggg");
			for (int i=0;i<45;i++) {
				if (i%9 == 0 || i%9 == 8) continue;
				show.setItem(i, new ItemStack(Material.RED_ROSE));
			}
			event.getPlayer().openInventory(show);
		} else if (event.getMaterial() == Material.STONE_HOE) {
			Inventory inv = event.getPlayer().getInventory();
			for (int i=0;i<10000;i++) {
				try {
					ItemStack display = new ItemStack(Material.THIN_GLASS, 1, (short) 8);
					ItemMeta meta = display.getItemMeta();
					meta.setDisplayName("Slot #" + i);
					display.setItemMeta(meta);
					inv.setItem(i, display);
				} catch (Exception e) {}
			}
		}
	}
}
