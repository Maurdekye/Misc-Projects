package mainpackage;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

import org.bukkit.Bukkit;
import org.bukkit.Effect;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityDamageEvent.DamageCause;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.BlockIterator;
import org.bukkit.util.Vector;

public class Mainfile extends JavaPlugin implements Listener {

	public static HashMap<String, Boolean> hasJump = new HashMap<String, Boolean>();
	public static HashMap<String, Boolean> flingReady = new HashMap<String, Boolean>();
	public static List<String> classes = Arrays.asList("Warrior", "Rogue",
			"Wizard", "Preist", "ACROBAT");

	@SuppressWarnings("deprecation")
	public void BreakSurround(final Block toBreak, final Material original,
			final Location originalLocation) {
		if (toBreak.getType() != original)
			return;
		if (toBreak.getLocation().distance(originalLocation) > 100)
			return;
		toBreak.getWorld().playEffect(toBreak.getLocation(), Effect.STEP_SOUND,
				toBreak.getTypeId());
		toBreak.breakNaturally();
		getServer().getScheduler().scheduleSyncDelayedTask(this,
				new Runnable() {
					public void run() {
						for (BlockFace bf : BlockFace.values())
							BreakSurround(toBreak.getRelative(bf), original,
									originalLocation);
					}
				}, 1);
	}

	public void onEnable() {
		for (Player ply : Bukkit.getOnlinePlayers()) {
			hasJump.put(ply.getName(), false);
			flingReady.put(ply.getName(), false);
		}

		saveConfig();
		getServer().getPluginManager().registerEvents(this, this);
		getServer().getScheduler().scheduleSyncRepeatingTask(this,
				new Runnable() {
					public synchronized void run() {
						for (Player ply : Bukkit.getOnlinePlayers()) {
							if (((Entity) ply).isOnGround()) {
								hasJump.put(ply.getName(), true);
								flingReady.put(ply.getName(), false);
								ply.setAllowFlight(true);
							} else if (ply.isFlying()
									&& ply.getGameMode() != GameMode.CREATIVE) {
								ply.setFlying(false);
								ply.setAllowFlight(false);
								Vector plyDir = ply.getVelocity();
								ply.setVelocity(new Vector(plyDir.getX(), 0.8,
										plyDir.getZ()));
								for (int i = 0; i < 3; i++)
									ply.getWorld()
											.playEffect(
													ply.getLocation()
															.add(new Vector(
																	Math.random() * 1 - 0.5,
																	0,
																	Math.random() * 1 - 0.5)),
													Effect.STEP_SOUND, 130);
								ply.setFallDistance(0);
								flingReady.put(ply.getName(), true);
							} else if (ply.isSneaking()
									&& hasJump.get(ply.getName())
									&& flingReady.get(ply.getName())) {
								hasJump.put(ply.getName(), false);
								Vector plyDir = ply.getLocation()
										.getDirection();
								ply.setVelocity(new Vector(plyDir.getX(), 0.2,
										plyDir.getZ()));
								for (int i = 0; i < 3; i++)
									ply.getWorld()
											.playEffect(
													ply.getLocation()
															.add(new Vector(
																	Math.random() * 1 - 0.5,
																	0,
																	Math.random() * 1 - 0.5)),
													Effect.STEP_SOUND, 152);
								ply.setFallDistance(0);
								flingReady.put(ply.getName(), false);
							}
						}
					}
				}, 0, 1);
	}

	@EventHandler
	public void Join(PlayerJoinEvent event) {
		Player ply = event.getPlayer();
		hasJump.put(ply.getName(), false);
		ArrayList<ItemStack> items = new ArrayList<ItemStack>();
		for (int i = 0; i < 5; i++) {
			ItemStack add = new ItemStack(Material.WOOL, 1, (short) i);
			ItemMeta meta = add.getItemMeta();
			meta.setDisplayName(ply.getName());
			add.setItemMeta(meta);
			add.addUnsafeEnchantment(Enchantment.DAMAGE_ALL, 5);
			items.add(add);
		}
		getConfig().set("items", items);
		saveConfig();
	}

	@EventHandler
	public void Click(PlayerInteractEvent event) {
		Player ply = event.getPlayer();
		if (ply.getItemInHand().getType() == Material.WATCH) {
			Inventory gui = Bukkit.createInventory(ply, 9, "Pick Your Class!");
			int i = 0;
			for (String classname : classes) {
				ItemStack item = new ItemStack(Material.SKULL_ITEM, 1,
						(short) i);
				ItemMeta meta = item.getItemMeta();
				meta.setDisplayName(classname);
				meta.setLore(Arrays.asList(classname));
				item.setItemMeta(meta);
				gui.addItem(item);
				gui.addItem(new ItemStack(Material.GLASS, 64));
				i++;
			}
			while (gui.contains(new ItemStack(Material.GLASS, 64)))
				gui.removeItem(new ItemStack(Material.GLASS, 64));
			ply.openInventory(gui);
		}
	}

	@EventHandler
	public void InvClick(InventoryClickEvent event) {
		if (!event.getInventory().getName().equals("Pick Your Class!"))
			return;
		event.setCancelled(true);
		event.getWhoClicked().setItemOnCursor(null);
		if (event.getCurrentItem() == null)
			return;
		if (event.getCurrentItem().getItemMeta() == null)
			return;
		List<String> lore = event.getCurrentItem().getItemMeta().getLore();
		if (lore == null)
			return;
		if (lore.size() == 0)
			return;
		if (!classes.contains(lore.get(0)))
			return;
		event.getWhoClicked().closeInventory();
		if (event.getWhoClicked() instanceof Player) {
			((Player) event.getWhoClicked())
					.sendMessage("You have chosen class "
							+ event.getCurrentItem().getItemMeta()
									.getDisplayName() + "!");
			@SuppressWarnings("unchecked")
			ArrayList<ItemStack> configs = (ArrayList<ItemStack>) getConfig()
					.getList("items");
			event.getWhoClicked()
					.getWorld()
					.dropItem(event.getWhoClicked().getEyeLocation(),
							configs.get((int) (Math.random() * configs.size())))
					.setVelocity(new Vector(0, 0, 0));
			;
		}
	}

	@EventHandler
	public void Splat(EntityDamageEvent event) {
		if (event.getEntity() instanceof Player) {
			Player ply = (Player) event.getEntity();
			if (event.getCause() == DamageCause.FALL) {
				if (flingReady.get(ply.getName()))
					event.setCancelled(true);
				if (!hasJump.get(ply.getName()))
					event.setCancelled(true);
			}
		}
	}

	@EventHandler
	public void interact(PlayerInteractEvent event) {
		if (event.getItem() == null)
			return;
		if (event.getItem().getType() == Material.BAKED_POTATO) {
			BlockIterator bi = new BlockIterator(event.getPlayer(), 100);
			while (bi.hasNext()) {
				Block b = bi.next();
				if (b.getType() == Material.AIR)
					continue;
				BreakSurround(b, b.getType(), b.getLocation());
				break;
			}
		}
	}
}


