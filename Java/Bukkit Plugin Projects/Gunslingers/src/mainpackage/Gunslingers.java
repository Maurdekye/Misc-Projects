package mainpackage;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import org.bukkit.ChatColor;
import org.bukkit.Effect;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.Sound;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Arrow;
import org.bukkit.entity.Entity;
import org.bukkit.entity.LivingEntity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.ExplosionPrimeEvent;
import org.bukkit.event.entity.ProjectileHitEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerItemHeldEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.metadata.FixedMetadataValue;
import org.bukkit.metadata.MetadataValue;
import org.bukkit.plugin.Plugin;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;
import org.bukkit.util.BlockIterator;
import org.bukkit.util.Vector;

public class Gunslingers extends JavaPlugin{
	
	public static String badPermissionsMessage = ChatColor.RED + "You don't have permission to use that command.";
	public static String badInterfaceMessage = ChatColor.RED + "You must be an active player to use that command.";
	public static boolean debugEnabled = false;
	public static boolean creativeOnly = false;
	public static boolean explosiveAerial = false;
	public static boolean infiniteAmmo = false;
	public static boolean noToolDamage = false;
	public static HashMap<String, Double> ranges = new  HashMap<>();
	public Plugin plug = this;
	public static int clock = 0;

	public static ArrayList<Material> swords = new ArrayList<>();
	public static ArrayList<String> swordBinds = new ArrayList<>();
	public static String[] weapon_names = {
		"shotgun",
		"burst",
		"sniper",
		"minigun",
		"launcher",
		"airstrike",
		"flamethrower",
		"dart",
		"cfour",
		"mortar",
		"prox_mine",
		"fire_mine"
	};
	public static HashMap<String, Material> weapon_items = new HashMap<String, Material>();
	public static HashMap<String, Material> weapon_ammos = new HashMap<String, Material>();
	public static HashMap<String, Integer> weapon_usage = new HashMap<String, Integer>();
	public static HashMap<String, Boolean> weapon_enables = new HashMap<String, Boolean>();
	public static HashMap<String, Double> weapon_durability =  new HashMap<String, Double>();
	public static HashMap<String, Integer> weapon_delay =  new HashMap<String, Integer>();
	
	public CustomConfig numBinds;
	public CustomConfig itemBinds;
	
	public static HashMap<String, HashMap<String, Integer>> timers = new HashMap<String, HashMap<String, Integer>>();
	public static HashMap<String, HashMap<String, Integer>> durabilities = new HashMap<String, HashMap<String, Integer>>();
	public static HashMap<String, HashMap<String, Location>> targets = new HashMap<String, HashMap<String, Location>>();
	public static HashMap<String, HashMap<String, Vector>> directs = new HashMap<String, HashMap<String, Vector>>();
	
	static ArrayList<Arrow> rockets = new ArrayList<Arrow>();
	static ArrayList<Arrow> cfours = new ArrayList<Arrow>();
	static ArrayList<Mine> mines = new ArrayList<Mine>();
	
	public static void register(Player p, Plugin plug) {
		timers.put(p.getName(), new HashMap<String, Integer>());
		durabilities.put(p.getName(), new HashMap<String, Integer>());
		targets.put(p.getName(), new HashMap<String, Location>());
		directs.put(p.getName(), new HashMap<String, Vector>());
		for (String wepname : weapon_names) {
			timers.get(p.getName()).put(wepname, 0);
			durabilities.get(p.getName()).put(wepname, 0);
			targets.get(p.getName()).put(wepname, p.getLocation());
			directs.get(p.getName()).put(wepname, p.getLocation().getDirection());
		}
		p.setMetadata("gunsActive", new FixedMetadataValue(plug, true));
	}
	
	public static void debug(String message) {
		if (debugEnabled) System.out.println(message);
	}
	
	/*public static boolean invHasMat(Inventory inv, Material mat) {
		for (ItemStack item : inv.getContents()) {
			if (item == null) continue;
			if (item.getType() == mat) return true;
		}
		return false;
	}*/
	
	public static Block getTarget(Player ply, int range) {
		BlockIterator sight = new BlockIterator(ply, range);
		while (sight.hasNext()) {
			Block curblock = sight.next();
			if (curblock.getType() == Material.AIR) continue;
			return curblock;
		}
		return null;
	}

	public static Block getSide(Player ply, int range) {
		BlockIterator sight = new BlockIterator(ply, range);
		Block lastblock = null;
		while (sight.hasNext()) {
			Block curblock = sight.next();
			if (curblock.getType() == Material.AIR) {
				lastblock = curblock;
			} else {
				return lastblock;
			}
		}
		return null;
	}
	
	/*public static boolean strArrayListContains(ArrayList<String> checkIn, String checkFor) {
		for (String checkstr : checkIn)
			if (checkstr.equals(checkFor)) return true;
		return false;
	}*/
	
	public static int useItem(Player ply, Material itemtoget, int needed) {
		if (ply.getGameMode() == GameMode.CREATIVE) return needed;
		if (infiniteAmmo) return needed;
		int total = 0;
		for (ItemStack item : ply.getInventory().getContents()) {
			if (item == null) continue;
			if (item.getType() == itemtoget) {
				if (item.getAmount() > needed-total) {
					item.setAmount(item.getAmount()-(needed-total));
					total = needed;
					break;
				} else {
					total += item.getAmount();
					ply.getInventory().removeItem(new ItemStack(itemtoget, item.getAmount()));
				}
			}
		}
		return total;
	}
	
	public static int howMuch(Player ply, Material check) {
		int total = 0;
		for (ItemStack item : ply.getInventory().getContents()) {
			if (item == null) continue;
			if (item.getType() != check) continue;
			total += item.getAmount();
		}
		return total;
	}
	
	public static boolean takeOnlyIf(Player ply, Material item, int needed) {
		if (ply.getGameMode() == GameMode.CREATIVE) return true;
		if (howMuch(ply, item) >= needed) {
			useItem(ply, item, needed);
			return true;
		}
		return false;
	}
	
	public static Arrow playerShootArrow(Player ply, float velocity, float spread, String type, Plugin plug) {
		return playerShootArrow(ply, ply.getEyeLocation().add(ply.getLocation().getDirection()), ply.getLocation().getDirection(), velocity, spread, type, plug);
	}

	public static Arrow playerShootArrow(Player ply, Vector direction, float velocity, float spread, String type, Plugin plug) {
		return playerShootArrow(ply, ply.getEyeLocation().add(ply.getLocation().getDirection()), direction, velocity, spread, type, plug);
	}

	public static Arrow playerShootArrow(Player ply, Location origin, Vector direction, float velocity, float spread, String type, Plugin plug) {
		Arrow arrow = ply.getWorld().spawnArrow(origin, direction, velocity, spread);
		arrow.setShooter(ply);
		arrow.setMetadata("DamageType", new FixedMetadataValue(plug, type));
		return arrow;
	}
	
	public static String getArrowType(Arrow ent, Plugin plug) {
		  List<MetadataValue> values = ent.getMetadata("DamageType");
		  if (values.size() == 0) return "none";
		  return values.get(0).asString();
	}
	
	public static boolean damage(Player owner, ItemStack item, String wepname) {
		if (owner.getGameMode() == GameMode.CREATIVE) return false;
		if (noToolDamage) return false;
		double amount = weapon_durability.get(wepname);
		if (amount <= 0) return false;
		if (wepname.equals("minigun")) {
			HashMap<String, Integer> playDur = durabilities.get(owner.getName());
			playDur.put(wepname, playDur.get(wepname) + 1);
			if (playDur.get(wepname) >= 32) {
				playDur.put(wepname, 0);
				item.setDurability((short) (item.getDurability() + (32*amount)));
			}
		} else if (amount >= 1) {
			item.setDurability((short) (item.getDurability() + amount));
		} else {
			HashMap<String, Integer> playDur = durabilities.get(owner.getName());
			playDur.put(wepname, playDur.get(wepname) + 1);
			if (playDur.get(wepname) >= 1.0/amount) {
				playDur.put(wepname, 0);
				item.setDurability((short) (item.getDurability() + 1));
			}
		}
		if (item.getDurability() > item.getType().getMaxDurability()) {
			item.setDurability((short) (item.getType().getMaxDurability()));
			owner.getInventory().remove(item);
			item.setType(Material.AIR);
			owner.getWorld().playSound(owner.getEyeLocation(), Sound.ENTITY_ITEM_BREAK, 1, 1);
			return true;
		}
		return false;
	}
	
	public void onEnable() {
		
		// Config assignments
		
		numBinds = new CustomConfig(this, "number_binds.yml");
		numBinds.getConfig().options().copyDefaults(true);
		numBinds.saveConfig();
		itemBinds = new CustomConfig(this, "item_binds.yml");
		itemBinds.getConfig().options().copyDefaults(true);
		itemBinds.saveConfig();
		getConfig().options().copyDefaults(true);
		saveConfig();
		
		debugEnabled = plug.getConfig().getBoolean("enable_debug_text");
		creativeOnly = plug.getConfig().getBoolean("creative_only");
		explosiveAerial = plug.getConfig().getBoolean("explosive_aerial");
		infiniteAmmo = plug.getConfig().getBoolean("infinite_ammo");
		noToolDamage = plug.getConfig().getBoolean("no_tool_damage");
		ranges.put("prox_mine", plug.getConfig().getDouble("prox_mine_range"));
		ranges.put("fire_mine", plug.getConfig().getDouble("fire_mine_range"));
		final int c_mortarDensity = plug.getConfig().getInt("mortar_density");
		final int c_airstrikeDensity = plug.getConfig().getInt("airstrike_density");
		final boolean noPerms = plug.getConfig().getBoolean("permissions_disabled");
		
		swords.add(Material.DIAMOND_SWORD);
		swords.add(Material.GOLD_SWORD);
		swords.add(Material.IRON_SWORD);
		swords.add(Material.STONE_SWORD);
		swords.add(Material.WOOD_SWORD);
		
		// Default weapon items
		weapon_items.put("shotgun", Material.DIAMOND_AXE);
		weapon_items.put("burst", Material.DIAMOND_PICKAXE);
		weapon_items.put("sniper", Material.DIAMOND_HOE);
		weapon_items.put("cfour", Material.GOLD_SPADE);
		weapon_items.put("dart", Material.GOLD_HOE);
		weapon_items.put("mortar", Material.GOLD_AXE);
		weapon_items.put("minigun", Material.DIAMOND_SWORD);
		weapon_items.put("launcher", Material.DIAMOND_HOE);
		weapon_items.put("airstrike", Material.GOLD_PICKAXE);
		weapon_items.put("flamethrower", Material.GOLD_SWORD);
		weapon_items.put("prox_mine", Material.FIREWORK_CHARGE);
		weapon_items.put("fire_mine", Material.BLAZE_ROD);
		
		// Default weapon ammos
		weapon_ammos.put("shotugn", Material.FLINT);
		weapon_ammos.put("burst", Material.STICK);
		weapon_ammos.put("sniper", Material.IRON_INGOT);
		weapon_ammos.put("cfour", Material.SULPHUR);
		weapon_ammos.put("dart", Material.GOLD_INGOT);
		weapon_ammos.put("mortar", Material.FEATHER);
		weapon_ammos.put("minigun", Material.SEEDS);
		weapon_ammos.put("launcher", Material.TNT);
		weapon_ammos.put("airstrike", Material.SUGAR_CANE);
		weapon_ammos.put("flamethrower", Material.GLOWSTONE_DUST);
		weapon_ammos.put("prox_mine", Material.FIREWORK_CHARGE);
		weapon_ammos.put("fire_mine", Material.BLAZE_ROD);
		
		// Default weapon usages
		weapon_usage.put("shotgun", 8);
		weapon_usage.put("burst", 1);
		weapon_usage.put("sniper", 1);
		weapon_usage.put("cfour", 16);
		weapon_usage.put("dart", 1);
		weapon_usage.put("mortar", 1);
		weapon_usage.put("minigun", 1);
		weapon_usage.put("launcher", 1);
		weapon_usage.put("airstrike", 1);
		weapon_usage.put("flamethrower", 1);
		weapon_usage.put("prox_mine", 8);
		weapon_usage.put("fire_mine", 8);
		
		// Default weapon durability usages
		weapon_durability.put("shotgun", 8.0);
		weapon_durability.put("burst", 8.0);
		weapon_durability.put("sniper", 16.0);
		weapon_durability.put("minigun", 1.0);
		weapon_durability.put("launcher", 32.0);
		weapon_durability.put("airstrike", 2.0);
		weapon_durability.put("flamethrower", 0.03125);
		weapon_durability.put("dart", 1.0);
		weapon_durability.put("cfour", 0.5);
		weapon_durability.put("mortar", 2.0);
		weapon_durability.put("prox_mine", 1.0);
		weapon_durability.put("fire_mine", 1.0);
		
		// Default weapon usage delays
		weapon_delay.put("shotgun", 50);
		weapon_delay.put("burst", 9);
		weapon_delay.put("sniper", 45);
		weapon_delay.put("cfour", 20);
		weapon_delay.put("dart", 90);
		weapon_delay.put("mortar", 300);
		weapon_delay.put("minigun", 0);
		weapon_delay.put("launcher", 250);
		weapon_delay.put("airstrike", 300);
		weapon_delay.put("flamethrower", 0);
		weapon_delay.put("prox_mine", 35);
		weapon_delay.put("fire_mine", 25);
		
		for (String wname : weapon_names) {
			
			// Assignment of items
			Material item = Material.getMaterial(itemBinds.getConfig().getString(wname));
			if (!itemBinds.getConfig().contains(wname)) {
				System.out.println("\tConfig Error: No entry in config for " + wname);
			} else if (item == null) {
				System.out.println("\tConfig Error: Bad item name for " + wname + "; " + itemBinds.getConfig().getString(wname) + " is not an item.");
			} else if (item.isBlock()) {
				System.out.println("\tConfig Error: Cannot use blocks as weapons.");
			} else if ((wname.equals("flamethrower") || wname.equals("minigun")) && !swords.contains(item)) {
				System.out.println("\tConfig Error: The " + wname + " must be bound to a sword.");
			} else {
				weapon_items.put(wname, item);
			}
			
			// Assignment of ammos
			Material ammo = Material.getMaterial(itemBinds.getConfig().getString(wname + "_ammo"));
			if (!itemBinds.getConfig().contains(wname + "_ammo")) {
				System.out.println("\tConfig Error: No entry in config for ammo of " + wname + ".");
			} else if (ammo == null) {
				System.out.println("\tConfig Error: Bad ammo name for " + wname + "; " + itemBinds.getConfig().getString(wname + "_ammo") + " is not an item.");
			} else {
				weapon_ammos.put(wname, ammo);
			}
			
			// Assignment of usages
			if (numBinds.getConfig().contains(wname + "_usage"))
				weapon_usage.put(wname, Math.abs(numBinds.getConfig().getInt(wname + "_usage")));
			
			// Assignment of enabled weapons
			if (getConfig().contains(wname + "_enabled"))
				weapon_enables.put(wname, getConfig().getBoolean(wname + "_enabled"));
			else
				weapon_enables.put(wname, true);
			
			// Assignment of durability usage
			if (numBinds.getConfig().contains(wname + "_durability"))
				weapon_durability.put(wname, numBinds.getConfig().getDouble(wname + "_durability"));
			
			// Assignment of weapon usage delays
			if (numBinds.getConfig().contains(wname + "_delay"))
				weapon_delay.put(wname, numBinds.getConfig().getInt(wname + "_delay"));
		}
		
		// Schedulers
		
		getServer().getPluginManager().registerEvents(new listen(this), this);
		getServer().getScheduler().scheduleSyncRepeatingTask(this, new Runnable () {
			public synchronized void run() {

				clock++;

				int mortarDensity = c_mortarDensity;
				int airstrikeDensity = c_airstrikeDensity;
				if (explosiveAerial) {
					mortarDensity /= 8;
					airstrikeDensity /= 8;
				}

				// Rocket Launcher
				if (weapon_enables.get("launcher")) {
					for (Arrow rocket : rockets) {
						rocket.getWorld().playEffect(rocket.getLocation(), Effect.SMOKE, 10);
					}
				}

				// C4
				if (weapon_enables.get("cfour")) {
					if (clock % 20 == 0) {
						for (Arrow cfour : cfours) {
							if (cfour.isOnGround()) cfour.getWorld().playEffect(cfour.getLocation(), Effect.SMOKE, 2);
						}
					}
				}

				// Mines
				if (weapon_enables.get("prox_mine") || weapon_enables.get("fire_mine")) {
					if (clock % 25 == 0) {
						for (Mine mine : mines) {
							mine.loc.getWorld().playEffect(mine.loc, Effect.SMOKE, 2);
						}
					}
				}

				for (String ply_s : timers.keySet()) {
					Player ply = getServer().getPlayer(ply_s);
					if (!ply.getMetadata("gunsActive").get(0).asBoolean()) continue;
					if (creativeOnly && ply.getGameMode() != GameMode.CREATIVE) continue;
					HashMap<String, Integer> timer = timers.get(ply.getName());
					boolean dead =  false;
					ItemStack hand = ply.getInventory().getItemInMainHand();

					// Minigun
					if (hand.getType() == weapon_items.get("minigun")) {
						if (!weapon_enables.get("minigun")) {debug("\tdisabled"); continue;}
						if (!ply.isBlocking()) continue;
						debug(ply.getName() + " using minigun.");
						if (timer.get("minigun") > 0) {debug("\tcooldown hot"); continue;}
						if (!noPerms && !ply.hasPermission("gunslingers.minigun")) {debug("\tno perms");continue;}
						if (!takeOnlyIf(ply, weapon_ammos.get("minigun"), weapon_usage.get("minigun"))) {debug("\tno ammo"); continue;}
						dead = damage(ply, hand, "minigun");
						timer.put("minigun", weapon_delay.get("minigun"));
						ply.getWorld().playSound(ply.getEyeLocation(), Sound.BLOCK_PISTON_CONTRACT, 0.5F, 10);
						ply.setVelocity(ply.getVelocity().subtract(ply.getLocation().getDirection().multiply(0.01)));
						playerShootArrow(ply, 3, 6, "minigun", plug);
					}

					// Flamethrower
					if (hand.getType() == weapon_items.get("flamethrower")) {
						if (!weapon_enables.get("flamethrower")) {debug("\tdisabled"); continue;}
						if (!ply.isBlocking()) continue;
						debug(ply.getName() + " using flamethrower.");
						if (!noPerms && !ply.hasPermission("gunslingers.flamethrower")) {debug("\tno perms"); continue;}
						if (timer.get("flamethrower") > 0) {debug("\tcooldown hot"); return;}
						if (ply.getWorld().hasStorm()) {
							ply.getWorld().playEffect(ply.getEyeLocation(), Effect.EXTINGUISH, 4);
							System.out.print("\tduring storm");
							continue;
						}
						if (!takeOnlyIf(ply, weapon_ammos.get("flamethrower"), weapon_usage.get("flamethrower"))) {debug("\tno ammo"); continue;}

						timer.put("minigun", weapon_delay.get("minigun"));
						dead = damage(ply, hand, "flamethrower");
						ply.getWorld().playEffect(ply.getLocation(), Effect.BLAZE_SHOOT, 2);
						ply.getWorld().playEffect(ply.getEyeLocation().add(ply.getLocation().getDirection()), Effect.SMOKE, 2);
						final Arrow arrow = playerShootArrow(ply, 3, 4, "flamethrower", plug);
						arrow.setFireTicks(100);
						plug.getServer().getScheduler().scheduleSyncDelayedTask(plug, new Runnable() {
							public void run() {
								arrow.remove();
							}
						}, 6);
					}

					// Burst Fire Gun
					if (timer.get("burst")%2 == 1 && timer.get("burst") >= weapon_delay.get("burst")) {
						if (takeOnlyIf(ply, weapon_ammos.get("burst"), weapon_usage.get("burst"))) {
							ply.getWorld().playSound(ply.getLocation(), Sound.ITEM_FLINTANDSTEEL_USE, (float) Math.random()*3+3, (float) Math.random()*3+3);
							playerShootArrow(ply, 3.5F, 1.25F, "burst", plug);
							debug("\tfire, " + timer.get("burst"));
						} else {
							timer.put("burst", weapon_delay.get("burst"));
							debug("\tno ammo");
						}
					} else if (ply.getGameMode() == GameMode.CREATIVE && timer.get("burst") < weapon_delay.get("burst")) {
						timer.put("burst", 1);
					}

					// Mortar Strike Designator
					if (timer.get("mortar") > weapon_delay.get("mortar")) {
						if (takeOnlyIf(ply, weapon_ammos.get("mortar"), weapon_usage.get("mortar"))) {
							for (int e=1;e<=mortarDensity;e++) {
								ply.getWorld().playSound(targets.get(ply.getName()).get("mortar"), Sound.ENTITY_GHAST_SHOOT, (float) Math.random()+1, (float) Math.random()+1);
								Location loc = null;
								Location orig = targets.get(ply.getName()).get("mortar").clone();
								while (loc == null || loc.distance(orig) > c_mortarDensity) {
									loc = targets.get(ply.getName()).get("mortar").clone();
									loc.setX(loc.getX()+((Math.random()*c_mortarDensity*2) - c_mortarDensity));
									loc.setZ(loc.getZ()+((Math.random()*c_mortarDensity*2) - c_mortarDensity));
								}
								ply.getWorld().createExplosion(loc, 0);
								playerShootArrow(ply, loc, new Vector(0, -1, 0), 4F, 12F, "mortar", plug);
							}
						} else {
							timer.put("mortar", weapon_delay.get("mortar"));
							System.out.print("\tno ammo");
						}
					} else if (ply.getGameMode() == GameMode.CREATIVE) {
						timer.put("mortar", 1);
					} else if (timer.get("mortar") == 1) {
 						ply.getWorld().playSound(ply.getEyeLocation(), Sound.ENTITY_PLAYER_LEVELUP, 4, 4);
					}

					// Airstrike Designator
					if (timer.get("airstrike") > weapon_delay.get("airstrike")) {
						if (takeOnlyIf(ply, weapon_ammos.get("airstrike"), weapon_usage.get("airstrike"))) {
							for (int e=1;e<=airstrikeDensity;e++) {
								ply.getWorld().playSound(targets.get(ply.getName()).get("airstrike"), Sound.ENTITY_GHAST_SHOOT, (float) Math.random()+1, (float) Math.random()+1);
								Location loc = targets.get(ply.getName()).get("airstrike").clone();
								loc.setX(loc.getX()+(Math.random()*2*c_airstrikeDensity - c_airstrikeDensity));
								loc.setZ(loc.getZ()+(Math.random()*2*c_airstrikeDensity - c_airstrikeDensity));
								ply.getWorld().createExplosion(loc, 0);
								playerShootArrow(ply, loc, new Vector(0, -1, 0), 4F, c_airstrikeDensity*2, "airstrike", plug);
							}
							HashMap<String, Location> locs = targets.get(ply.getName());
							locs.get("airstrike").setX(locs.get("airstrike").getX()+(directs.get(ply.getName()).get("airstrike").getX()*0.5));
							locs.get("airstrike").setZ(locs.get("airstrike").getZ()+(directs.get(ply.getName()).get("airstrike").getZ()*0.5));
							targets.put(ply_s, locs);
						} else {
							timer.put("airstrike", weapon_delay.get("airstrike"));
							System.out.print("\tno ammo");
						}
					} else if (ply.getGameMode() == GameMode.CREATIVE) {
						timer.put("airstrike", 1);
					} else if (timer.get("airstrike") == 1) {
						ply.getWorld().playSound(ply.getEyeLocation(), Sound.ENTITY_FIREWORK_TWINKLE, 2, 8);
					}

					if (timer.get("shotgun") == 1) {
						ply.playSound(ply.getEyeLocation(), Sound.BLOCK_WOODEN_DOOR_CLOSE, 2, 10F);
					}

					if (timer.get("launcher") == 3) ply.playSound(ply.getEyeLocation(), Sound.BLOCK_PISTON_EXTEND, 2, 1.5F);
					if (timer.get("launcher") == 1) ply.playSound(ply.getEyeLocation(), Sound.BLOCK_PISTON_CONTRACT, 2, 1.2F);


					for (String wep : timer.keySet()) {
						timer.put(wep, Math.max(timer.get(wep) - 1, 0));
					}

					if (dead) ply.getWorld().playSound(ply.getEyeLocation(), Sound.ENTITY_ITEM_BREAK, 1, 1);
					timers.put(ply_s, timer);
				}
			}
		}, 0, 1);
		for (World w : getServer().getWorlds()) {
			for (Player p : w.getPlayers()) {
				register(p, plug);
			}
		}
	}
	
	public void onDisable() {}
	
	/*public static boolean hasInt(int[] list, int lookfor) {
		for (int s : list) {if (s == lookfor) {return true;}}
		return false;
	}*/
	
	public boolean onCommand(CommandSender sender, Command cmd, String command, String[] args) {
		boolean isCmd = !(sender instanceof Player);
		boolean isOp = !isCmd && ((Player) sender).isOp();
		boolean hasPow = isCmd || isOp;
		
		if (cmd.getLabel().equalsIgnoreCase("togglegunsling")) {
			if (args.length > 0) {
				if (!hasPow) {
					sender.sendMessage("You may only activate / deactivate gunslinging for yourself.");
				} else {
					Player target = getServer().getPlayer(args[0]);
					if (target == null) {
						sender.sendMessage(ChatColor.RED + "Player '" + args[0] + "' is not currently on this server.");
					} else {
						boolean active = target.getMetadata("gunsActive").get(0).asBoolean();
						target.setMetadata("gunsActive", new FixedMetadataValue(plug, !active));
						String isOn = ChatColor.GREEN + "activated";
						if (active) isOn = ChatColor.RED + "deactivated";
						sender.sendMessage(ChatColor.GOLD + "Gunslingers has been " + isOn + ChatColor.GOLD + " for player " + target.getName() + '.');
					}
				}
			} else {
				if (!isCmd) {
					Player ply = (Player) sender;
					boolean active = ply.getMetadata("gunsActive").get(0).asBoolean();
					ply.setMetadata("gunsActive", new FixedMetadataValue(plug, !active));
					String isOn = ChatColor.GREEN + "activated";
					if (active) isOn = ChatColor.RED + "deactivated";
					ply.sendMessage(ChatColor.GOLD + "Gunslingers has been " + isOn + ChatColor.GOLD + '.');
				} else {
					sender.sendMessage(badInterfaceMessage);
				}
			}
		}
		
		else {
			sender.sendMessage("Command \"/' + command + '\" has not been implemented yet.");
		}
		return true;
	}
	
	
}

class listen implements Listener {
	
	public String multicolor(String instr) {
		String fin = "";
		ChatColor[] colors = ChatColor.values();
		for (int i=0;i<instr.length();i++) {
			fin = fin + colors[(int) (Math.random()*15)] + instr.charAt(i);
		}
		return fin;
	}

	public String locationToString(Location loc) {
		if (loc == null) return "null location";
		String fin = "(";
		fin += loc.getX() + ", ";
		fin += loc.getY() + ", ";
		fin += loc.getZ() + ")";
		fin += " in world " + loc.getWorld().getName();
		return fin;
	}
	
	public Gunslingers plug;

	listen(Gunslingers main) {
		this.plug = main;
	}
	
	@EventHandler
	public void Join(PlayerJoinEvent event) {
		Gunslingers.register(event.getPlayer(), plug);
	}
	
	@EventHandler
	public void Leave(PlayerQuitEvent event) {
		Gunslingers.timers.remove(event.getPlayer().getName());
	}
	
	@EventHandler
	public void Hotbar(PlayerItemHeldEvent event) {
		Material item = event.getPlayer().getInventory().getItemInMainHand().getType();
		if (item == Gunslingers.weapon_items.get("sniper") || item == Gunslingers.weapon_items.get("dart")) return;
		if (!event.getPlayer().hasPotionEffect(PotionEffectType.SLOW)) return;
		event.getPlayer().removePotionEffect(PotionEffectType.SLOW);
	}
	
	@EventHandler
	public void Break(BlockBreakEvent event) {
		ArrayList<Mine> removes = new ArrayList<Mine>();
		for (Mine mine : Gunslingers.mines) {
			if (mine.disable(event.getBlock())) removes.add(mine);
		}
		for (Mine m : removes) Gunslingers.mines.remove(m);
	}
	
	@EventHandler
	public void Explosion(ExplosionPrimeEvent event) {
		float dist = event.getRadius();
		ArrayList<Mine> explosions = new ArrayList<Mine>();
		for (final Mine mine : Gunslingers.mines) {
			if (mine.loc.distance(event.getEntity().getLocation()) <= dist) {
				explosions.add(mine);
			}
		}
		for (final Mine mine : explosions) {
			mine.detonate();
			Gunslingers.mines.remove(mine);
		}
	}
	  
	@EventHandler
	public void ArrowHit(ProjectileHitEvent event) {
	    if(!(event.getEntity() instanceof Arrow)) return;
	    Arrow arrow = (Arrow) event.getEntity();
	    String type = Gunslingers.getArrowType(arrow, plug);
	    if ((type.equals("flamethrower") && plug.getConfig().getBoolean("flamethrower_ignite_blocks")) || type.equals("fire_mine")) {
		    BlockIterator iterator = new BlockIterator(arrow.getWorld(), arrow.getLocation().toVector(), arrow.getVelocity().normalize().multiply(-1), 0, 4);
		    Block hitBlock = iterator.next();
		    while(iterator.hasNext()) {
		        hitBlock = iterator.next();
		        if(hitBlock.getType() == Material.AIR) break;
		    }
		    hitBlock.setType(Material.FIRE);
	    } else if ((type.equals("mortar") || type.equals("airstrike")) && Gunslingers.explosiveAerial) {
	    	arrow.getWorld().createExplosion(arrow.getLocation(), plug.getConfig().getInt("aerial_explosive_power"));
	    	arrow.remove();
	    } else if (type.equals("launcher")) {
			arrow.getWorld().createExplosion(arrow.getLocation(), plug.getConfig().getInt("launcher_explosive_power"));
			Gunslingers.rockets.remove(arrow);
			arrow.remove();
	    }
	}
	
	@EventHandler
	public void Move(PlayerMoveEvent event) {
		final Player ply = event.getPlayer();
		ArrayList<Mine> explosions = new ArrayList<Mine>();
		for (Mine mine : Gunslingers.mines) {
			if (mine.loc.getWorld() != ply.getWorld()) continue;
			if (mine.loc.distance(ply.getLocation()) <= Gunslingers.ranges.get(mine.type)) {
				explosions.add(mine);
			}
		}
		for (Mine mine : explosions) {
			mine.detonate();
			Gunslingers.mines.remove(mine);
		}
	}
	
	@EventHandler
	public void Damage(EntityDamageByEntityEvent event) {
		if (event.getDamager() instanceof Arrow) {
			Arrow arrow = (Arrow) event.getDamager();
			Entity hit = event.getEntity();
			String type = Gunslingers.getArrowType(arrow, plug);
			switch (type) {
				case "shotgun": event.setDamage(6.0);
					hit.setVelocity(hit.getVelocity().add(arrow.getVelocity().multiply(0.15)));
					break;
				case "sniper": event.setDamage(20.0);
					break;
				case "burst": event.setDamage(4.0);
					break;
				case "cfour": event.setDamage(1.0);
					break;
				case "dart": event.setDamage(1.0);
					if (event.getEntity() instanceof LivingEntity)
						((LivingEntity) event.getEntity()).addPotionEffect(new PotionEffect(PotionEffectType.POISON, 100, 2));
					break;
				case "mortar": event.setDamage(4.0);
					if (Gunslingers.explosiveAerial) {
						arrow.getWorld().createExplosion(arrow.getLocation(), plug.getConfig().getInt("aerial_explosive_power"));
						arrow.remove();
					}
					break;
				case "airstrike": event.setDamage(8.0);
					if (Gunslingers.explosiveAerial) {
						arrow.getWorld().createExplosion(arrow.getLocation(), plug.getConfig().getInt("aerial_explosive_power"));
						arrow.remove();
					}
					break;
				case "minigun": event.setDamage(1.5);
					break;
				case "launcher": event.setDamage(2.0);
					arrow.getWorld().createExplosion(arrow.getLocation(), plug.getConfig().getInt("launcher_explosive_power"));
					Gunslingers.rockets.remove(arrow);
					arrow.remove();
					break;
				case "flamethrower": event.setDamage(1.0);
					event.getEntity().setFireTicks(150);
					break;
				default: 
					break;
			}
		}
	}
    
	// ----------------- Gun Functionality ------------------- \\
	
	@SuppressWarnings("deprecation")
	@EventHandler
	public void Click(PlayerInteractEvent event) {
		
		final Player ply = event.getPlayer();
		boolean noPerms = plug.getConfig().getBoolean("permissions_disabled");
		if (!ply.getMetadata("gunsActive").get(0).asBoolean()) {Gunslingers.debug("\tplugin deactivated"); return;}
		if (Gunslingers.creativeOnly && ply.getGameMode() != GameMode.CREATIVE) return;
		String enme = event.getAction().toString();
		if (enme.equals("PHYSICAL")) return;
		ItemStack hand = ply.getItemInHand();
		if (!Gunslingers.timers.containsKey(ply.getName())) Gunslingers.register(ply, plug);
		HashMap<String, Integer> timer = Gunslingers.timers.get(ply.getName());
		boolean dead = false;
		boolean left = enme.equals("LEFT_CLICK_AIR") || enme.equals("LEFT_CLICK_BLOCK");
		ArrayList<String> weapons = new ArrayList<String>();
		
		for (final String wepname : Gunslingers.weapon_names) {
			if (hand.getType() == Gunslingers.weapon_items.get(wepname)) {
				weapons.add(wepname);
				Gunslingers.debug(ply.getName() + " used weapon " + wepname + ".");
				if (left) {
					if (timer.get(wepname) > 0)  {Gunslingers.debug("\tcooldown hot, " + timer.get(wepname)); continue;}
					if (!noPerms && !ply.hasPermission("gunslingers." + wepname)) {Gunslingers.debug("\tno perms"); continue;}
					if (!Gunslingers.weapon_enables.get(wepname)) {Gunslingers.debug("\tdisabled"); continue;}
					if (wepname.equals("fire_mine") || wepname.equals("prox_mine")) continue;
					if (!Gunslingers.takeOnlyIf(ply, Gunslingers.weapon_ammos.get(wepname), Gunslingers.weapon_usage.get(wepname))) {Gunslingers.debug("\tno ammo"); continue;}
					Block seeblock = Gunslingers.getTarget(ply, 128);
					if (seeblock != null)
						Gunslingers.targets.get(ply.getName()).put(wepname, seeblock.getLocation());
					Gunslingers.directs.get(ply.getName()).put(wepname, ply.getLocation().getDirection());
					if (ply.getGameMode() != GameMode.CREATIVE) timer.put(wepname, Gunslingers.weapon_delay.get(wepname));
					dead = Gunslingers.damage(ply, hand, wepname);
					
					// Shotgun
					if (wepname.equals("shotgun")) {
						ply.setVelocity(ply.getVelocity().subtract(ply.getLocation().getDirection().multiply(0.3)));
						ply.getWorld().playSound(ply.getLocation(), Sound.ENTITY_GENERIC_EXPLODE, ((float) 8), (float) Math.random() + 2);
						for (int i=0; i<8; i++) {
							Gunslingers.playerShootArrow(ply, 3, 15, "shotgun", plug);
						}	
					}
					
					// Burst
					else if (wepname.equals("burst")) {
						timer.put("burst", Gunslingers.weapon_delay.get("burst") + (plug.getConfig().getInt("burst_ticks") - 1)*2);
					}
					
					// Sniper Rifle / Dart Gun
					else if (wepname.equals("sniper") || wepname.equals("dart")) {
						ply.getWorld().playSound(ply.getLocation(), new Sound[]{Sound.ENTITY_FIREWORK_BLAST, Sound.ENTITY_FIREWORK_LARGE_BLAST}[(int) (Math.random()*2)], (float) Math.random()*2, (float) Math.random()*2);
						Vector addVec = new Vector(0, 0.02, 0);
						if (ply.hasPotionEffect(PotionEffectType.SLOW)) addVec.setY(0.04);
						Gunslingers.playerShootArrow(ply, ply.getLocation().getDirection().add(addVec), 4, 0.001F, wepname, plug);
					}
					
					// C4 Placer
					else if (wepname.equals("cfour")) {
						final Arrow arrow = Gunslingers.playerShootArrow(ply, ply.getLocation().getDirection().add(new Vector(0, 0.2F, 0)), 0.2F, 0F, "cfour", plug);
						ply.getWorld().playSound(ply.getLocation(), Sound.BLOCK_LAVA_POP, 5, 2);
						plug.getServer().getScheduler().scheduleSyncDelayedTask(plug, new Runnable() {
							public void run() {
								if (arrow.isValid()) {
									arrow.getWorld().createExplosion(arrow.getLocation(), 3);
									Gunslingers.cfours.remove(arrow);
									arrow.remove();
								}
							}
						}, 1190);
						Gunslingers.cfours.add(arrow);
					}
					
					// Rocket Launcherws
					else if (wepname.equals("launcher")) {
						ply.getWorld().playSound(ply.getEyeLocation(), Sound.ENTITY_FIREWORK_LAUNCH, 1, 20);
						Gunslingers.rockets.add(Gunslingers.playerShootArrow(ply, ply.getLocation().getDirection().add(new Vector(0, 0.15, 0)), 1.5F, 0.1F, "launcher", plug));
					}
					
					// Mortar / Airstrike
					else if (wepname.equals("mortar") || wepname.equals("airstrike")) {
						if (wepname.equals("mortar") && seeblock == null) {Gunslingers.debug("\tbad selection"); continue;}
						timer.put(wepname, Gunslingers.weapon_delay.get(wepname) + plug.getConfig().getInt(wepname + "_ticks"));
						if (wepname.equals("airstrike")) 
							Gunslingers.targets.get(ply.getName()).put(wepname, ply.getLocation().add(ply.getLocation().getDirection()));
						Gunslingers.targets.get(ply.getName()).get(wepname).add(new Vector(0, 40, 0));
						Gunslingers.debug("\tlocation of " + locationToString(Gunslingers.targets.get(ply.getName()).get(wepname)));
					}
					
				} else {
					
					// Rifle Zoom
					if (wepname.equals("sniper") || wepname.equals("dart")) {
						if (ply.hasPotionEffect(PotionEffectType.SLOW)) ply.removePotionEffect(PotionEffectType.SLOW);
						else ply.addPotionEffect(new PotionEffect(PotionEffectType.SLOW, 1200, 100));
						
					// C4 Detonate
					} else if (wepname.equals("cfour")) {
						for (final Arrow cfour : Gunslingers.cfours) {
							if (((Player) cfour.getShooter()).getName().equals(ply.getName()) && cfour.isValid()) {
								cfour.getWorld().playSound(cfour.getLocation(), Sound.BLOCK_LEVER_CLICK, 4, 10);
								cfour.getWorld().playEffect(cfour.getLocation(), Effect.MOBSPAWNER_FLAMES, 3);
								plug.getServer().getScheduler().scheduleSyncDelayedTask(plug, new Runnable() {
									public void run() {
										if (cfour.isValid()) {
											cfour.getWorld().createExplosion(cfour.getLocation(), plug.getConfig().getInt("cfour_explosive_power"));
											Gunslingers.cfours.remove(cfour);
											cfour.remove();
										}
									}
								}, 5);
							}
						}
						
					// Proximity Mine / Incendiary Mine
					} else if (wepname.equals("prox_mine") || wepname.equals("fire_mine")) {
						if (timer.get(wepname) > 0)  {Gunslingers.debug("\tcooldown hot, " + timer.get(wepname)); continue;}
						if (!noPerms && !ply.hasPermission("gunslingers." + wepname)) {Gunslingers.debug("\tno perms"); continue;}
						if (!Gunslingers.weapon_enables.get(wepname)) {Gunslingers.debug("\tdisabled"); continue;}
						if (!Gunslingers.takeOnlyIf(ply, Gunslingers.weapon_ammos.get(wepname), Gunslingers.weapon_usage.get(wepname))) {Gunslingers.debug("\tno ammo"); continue;}
						int range = 4;
						if (ply.getGameMode() == GameMode.CREATIVE) range = 128;
						final Block target = Gunslingers.getSide(ply, range);
						final Block look = Gunslingers.getTarget(ply, 128);
						if (target == null) {Gunslingers.debug("\tbad selection"); continue;}
						if (ply.getGameMode() != GameMode.CREATIVE) timer.put(wepname, Gunslingers.weapon_delay.get(wepname));
						int delay = 40;
						if (wepname.equals("prox_mine")) delay = plug.getConfig().getInt("prox_mine_ready_delay");
						if (wepname.equals("fire_mine")) delay = plug.getConfig().getInt("fire_mine_ready_delay");
						ply.getWorld().playEffect(look.getLocation(), Effect.STEP_SOUND, look.getTypeId());
						plug.getServer().getScheduler().scheduleSyncDelayedTask(plug, new Runnable() {
							public void run() {
								ply.getWorld().playEffect(ply.getLocation(), Effect.CLICK2, 1);
								Gunslingers.mines.add(new Mine(ply, target.getLocation(), look.getLocation(), wepname, plug));
							}
						}, delay);
					} else {
						Gunslingers.debug("\tRight Click");
					}
				}
			}
		}
		
		Gunslingers.debug("\n");
		
		if (dead) {
			if (weapons.contains("cfour") && plug.getConfig().getBoolean("cfour_explode_on_weapon_loss")) {
				for (int i=0;i<Gunslingers.cfours.size();i++) {
					final Arrow cfour = Gunslingers.cfours.get(i);
					if (((Player) cfour.getShooter()).getName().equals(ply.getName())) {
						cfour.getWorld().playSound(cfour.getLocation(), Sound.BLOCK_LEVER_CLICK, 4, 10);
						cfour.getWorld().playEffect(cfour.getLocation(), Effect.MOBSPAWNER_FLAMES, 3);
						plug.getServer().getScheduler().scheduleSyncDelayedTask(plug, new Runnable() {
							public void run() {
								if (cfour.isValid()) {
									cfour.getWorld().createExplosion(cfour.getLocation(), plug.getConfig().getInt("cfour_explosive_power"));
									Gunslingers.cfours.remove(cfour);
									cfour.remove();
								}
							}
						}, 5);
					}
				}
			}
		}
		
		Gunslingers.timers.put(ply.getName(), timer);
	}
	
}

class Mine {
	Player owner;
	Location loc;
	Location house;
	String type;
	Plugin plug;
	boolean dead;
	int power;

	Mine(Player owner, Location loc, Location house, String type, Plugin plug) {
		this.owner = owner;
		this.loc = loc;
		this.house = house;
		this.type = type;
		this.plug = plug;
		this.dead = false;
		this.power = plug.getConfig().getInt("fire_mine_explosive_power");
	}
	
	public Vector randomVector() {
		double x = Math.sin(Math.random()*Math.PI*2);
		double y = Math.random()*2 - 1;
		double z = Math.cos(Math.random()*Math.PI*2);
		return new Vector(x, y, z).normalize();
	}
	
	public void detonate() {
		if (this.dead) return;
		final Player ply = this.owner;
		this.owner.getWorld().playSound(loc, Sound.BLOCK_ANVIL_BREAK, 10, 18);
		Gunslingers.mines.remove(this);
		plug.getServer().getScheduler().scheduleSyncDelayedTask(plug, () -> {
            if (type.equals("prox_mine")) {
                loc.getWorld().createExplosion(loc, power);
            } else if (type.equals("fire_mine")) {
                loc.getWorld().createExplosion(loc, 0);
                for (int i=0;i<power*80;i++) {
                    final Arrow arrow = Gunslingers.playerShootArrow(ply, loc, randomVector(), 3, 1, type, plug);
                    arrow.setFireTicks(100);
                    plug.getServer().getScheduler().scheduleSyncDelayedTask(plug, arrow::remove, plug.getConfig().getInt("fire_mine_explosive_power"));
                }
            }
        }, 8);
	}
	
	public boolean disable(Block b) {
		if (b.getLocation().equals(house)) return true;
		this.dead = true;
		return false;
	}
}