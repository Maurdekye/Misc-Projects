package com.ncolaprete.magicaxe;

import org.bukkit.*;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.entity.Damageable;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.enchantment.EnchantItemEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerItemBreakEvent;
import org.bukkit.event.player.PlayerMoveEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.Vector;

import java.util.Arrays;
import java.util.HashMap;
import java.util.UUID;

public class MagicAxe extends JavaPlugin implements Listener, CommandExecutor
{
    private StateFlagManager Flags = new StateFlagManager();
    private HashMap<UUID, Integer> resetDelays = new HashMap<>();
    private HashMap<UUID, Integer> landDelays = new HashMap<>();
    private int Clock = 0;

    // config variables
    private int minimumXpCost;
    private float enchantChance;
    private short superjumpDurability;
    private short slamDurability;
    private short shortjumpDurability;
    private String axeLore;
    private boolean anyAxe;
    private boolean allEnchant;

    @Override
    public void onEnable()
    {
        getServer().getPluginManager().registerEvents(this, this);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::runOnTick, 0, 8);

        // set up config file
        FileConfiguration cfg = getConfig();
        cfg.addDefault("minimum_xp_cost", 10);
        cfg.addDefault("enchant_chance", 0.1f);
        cfg.addDefault("superjump_durability_usage", 2);
        cfg.addDefault("slam_durability_usage", 3);
        cfg.addDefault("shortjump_durability_usage", 1);
        cfg.addDefault("axe_lore", "A magical axe that allows you to jump great distances");
        cfg.addDefault("debug.any_axe", false);
        cfg.addDefault("debug.all_enchants_work", false);
        cfg.options().copyDefaults(true);
        saveConfig();

        // fetch config values
        reloadConfig();
        cfg = getConfig();
        minimumXpCost = cfg.getInt("minimum_xp_cost");
        enchantChance = (float) cfg.getDouble("enchant_chance");
        superjumpDurability = (short) cfg.getInt("superjump_durability_usage");
        slamDurability = (short) cfg.getInt("slam_durability_usage");
        shortjumpDurability = (short) cfg.getInt("shortjump_durability_usage");
        axeLore = cfg.getString("axe_lore");
        anyAxe = cfg.getBoolean("debug.any_axe");
        allEnchant = cfg.getBoolean("debug.all_enchants_work");
    }

    public boolean onCommand(CommandSender sender, Command command, String label, String[] args)
    {
        if (command.getName().equalsIgnoreCase("givemagicaxe") && sender instanceof Player && sender.hasPermission("magicaxe.givemagicaxe"))
        {
            Player ply = (Player) sender;
            ply.getInventory().addItem(getMagicAxe());
        }
        else if (command.getName().equalsIgnoreCase("maxdurability") && sender instanceof Player && sender.hasPermission("magicaxe.maxdurability"))
        {
            Player ply = (Player) sender;
            ItemStack handItem = ply.getInventory().getItemInMainHand();
            if (handItem != null)
                handItem.setDurability(Short.MIN_VALUE);
        }
        return true;
    }

    private void runOnTick()
    {
        Clock++;
        for (Player p : getServer().getOnlinePlayers())
        {
            //p.sendMessage(p.getLocation().getPitch() + "");
            //p.sendMessage(Flags.get(p, StateFlag.superLeap) + "" + Flags.get(p, StateFlag.slamming));
            //p.sendMessage(p.getHealth() + "");
        }
    }

    @EventHandler
    public void entityDamage(EntityDamageEvent e)
    {
        if (e.getEntity() instanceof Player && e.getCause() == EntityDamageEvent.DamageCause.FALL)
        {
            if (landDelays.containsKey(e.getEntity().getUniqueId()))
                getServer().getScheduler().cancelTask(landDelays.get(e.getEntity().getUniqueId()));
            boolean cancel = playerLand((Player)e.getEntity(), e.getDamage());
            e.setCancelled(cancel);
        }
    }

    @EventHandler
    public void playerMove(PlayerMoveEvent e)
    {
        boolean onGround = grounded(e.getPlayer());
        if (onGround && Flags.get(e.getPlayer(), StateFlag.inAir))
        {
            int taskId = getServer().getScheduler().scheduleSyncDelayedTask(this, () -> playerLand(e.getPlayer(), 0), 1);
            landDelays.put(e.getPlayer().getUniqueId(), taskId);
        }
        Flags.set(e.getPlayer(), StateFlag.inAir, !onGround);
    }

    private boolean playerLand(Player ply, double damage)
    {
        boolean cancel = false;
        if (Flags.get(ply, StateFlag.superLeap))
        {
            cancel = true;
            if (Flags.get(ply, StateFlag.slamming))
            {
                Flags.set(ply, StateFlag.slamming, false);
                ply.spawnParticle(Particle.BLOCK_DUST, ply.getLocation(), 50 * (int)damage, damage, 0, damage);
                double radius = Math.sqrt(damage * 10);
                int box = (int) radius;
                for (Entity ent : ply.getWorld().getNearbyEntities(ply.getLocation(), box, box, box))
                {
                    if (ent.getUniqueId() == ply.getUniqueId())
                        continue;
                    if (ent.getLocation().distance(ply.getLocation()) <= radius)
                    {
                        if (ent instanceof Damageable)
                        {
                            Vector direction = ent.getLocation().subtract(ply.getLocation()).toVector();
                            double magnitude = Math.min(damage, 1/direction.length() * damage);
                            direction = direction.normalize().add(new Vector(0, 0.5, 0));
                            ent.setVelocity(direction.multiply(magnitude/8f));
                            if (magnitude > 0.5)
                            {
                                ((Damageable) ent).damage(magnitude);
                                //ply.sendMessage("hit " + ent.getType().toString().toLowerCase() + " for " + Math.round(magnitude*100)/100.0f);
                            }
                        }
                    }
                }
                ply.playSound(ply.getLocation(), Sound.ENTITY_GENERIC_EXPLODE, 1, 1);
            }
            Flags.set(ply, StateFlag.superLeap, false);
            Flags.set(ply, StateFlag.slamming, false);
        }
        return cancel;
    }

    @EventHandler
    public void playerInteract(PlayerInteractEvent e)
    {
        Action ac = e.getAction();
        Player ply = e.getPlayer();
        boolean leftClick = ac == Action.LEFT_CLICK_AIR || ac == Action.LEFT_CLICK_BLOCK;
        boolean rightClick = ac == Action.RIGHT_CLICK_AIR || ac == Action.RIGHT_CLICK_BLOCK;
        ItemStack handItem = ply.getInventory().getItemInMainHand();
        boolean onGround = grounded(ply);
        if (leftClick || rightClick)
        {
            if (handItem.getType() == Material.GOLD_AXE &&
                    ((handItem.getItemMeta() != null &&
                    handItem.getItemMeta().getLore() != null &&
                    handItem.getItemMeta().getLore().size() > 0 &&
                    handItem.getItemMeta().getLore().get(0).equals(axeLore)) || anyAxe))
            {
                if (onGround)
                {
                    float multiplier = 4;
                    if (rightClick)
                        multiplier = 1.5f;
                    ply.setVelocity(ply.getLocation().getDirection().multiply(multiplier));
                    ply.playSound(ply.getLocation(), Sound.ENTITY_PLAYER_ATTACK_STRONG, 4, 1);
                    Flags.set(ply, StateFlag.superLeap, true);
                    if (resetDelays.containsKey(ply.getUniqueId()))
                        getServer().getScheduler().cancelTask(resetDelays.get(ply.getUniqueId()));
                    int taskId = getServer().getScheduler().scheduleSyncDelayedTask(this, () -> {
                        if (ply.isValid() && grounded(ply))
                            playerLand(ply, 0);
                    }, 5);
                    resetDelays.put(ply.getUniqueId(), taskId);
                    if (leftClick)
                        reduceDurability(ply, handItem, superjumpDurability);
                    else if (rightClick)
                        reduceDurability(ply, handItem, shortjumpDurability);
                }
                else if (Flags.get(ply, StateFlag.superLeap) && !Flags.get(ply, StateFlag.slamming) && ply.getLocation().getPitch() > 30 && !rightClick)
                {
                    ply.setVelocity(ply.getLocation().getDirection().multiply(4));
                    ply.playSound(ply.getLocation(), Sound.ENTITY_PLAYER_ATTACK_SWEEP, 4, 1);
                    Flags.set(ply, StateFlag.slamming, true);
                    reduceDurability(ply, handItem, slamDurability);
                }
            }
        }
    }

    @EventHandler
    private void enchantItem(EnchantItemEvent e)
    {
        if (e.getItem().getType() == Material.GOLD_AXE &&
                (allEnchant || (e.getExpLevelCost() > minimumXpCost && Math.random() < enchantChance)))
        {
            e.getInventory().setItem(0, getMagicAxe());
        }
    }

    private ItemStack getMagicAxe()
    {
        ItemStack magicAxe = new ItemStack(Material.GOLD_AXE);
        ItemMeta axeMeta = magicAxe.getItemMeta();
        axeMeta.setDisplayName(ChatColor.LIGHT_PURPLE + "Magical Axe");
        axeMeta.setLore(Arrays.asList(axeLore));
        magicAxe.setItemMeta(axeMeta);
        /*try
        {
            Field ISHandle = CraftItemStack.class.getDeclaredField("handle");
            ISHandle.setAccessible(true);
            net.minecraft.server.v1_9_R1.ItemStack nmsItem = (net.minecraft.server.v1_9_R1.ItemStack)ISHandle.get(magicAxe);
            NBTTagCompound tag = nmsItem.getTag();
            tag.set("ench", new NBTTagList());
            nmsItem.setTag(tag);
        }
        catch (Exception e) {}*/

        return magicAxe;
    }

    private boolean grounded(Player p)
    {
        return ((Entity) p).isOnGround();
    }

    private boolean reduceDurability(Player ply, ItemStack item, int amount)
    {
        if (ply.getGameMode() == GameMode.CREATIVE)
            return false;
        item.setDurability((short)(item.getDurability() + amount));
        if (item.getDurability() >= item.getType().getMaxDurability())
        {
            PlayerItemBreakEvent ev = new PlayerItemBreakEvent(ply, item);
            getServer().getPluginManager().callEvent(ev);
            ply.playSound(ply.getLocation(), Sound.ENTITY_ITEM_BREAK, 1, 1);
            ply.spawnParticle(Particle.ITEM_CRACK, ply.getLocation(), 20);
            ply.getInventory().remove(item);
            return true;
        }
        return false;
    }
}

class StateFlagManager
{
    private HashMap<UUID, HashMap<StateFlag, Boolean>> Flags;

    StateFlagManager()
    {
        Flags = new HashMap<>();
    }

    void set(Player ply, StateFlag flag, boolean value)
    {
        if (!Flags.containsKey(ply.getUniqueId()))
            Flags.put(ply.getUniqueId(), new HashMap<>());
        Flags.get(ply.getUniqueId()).put(flag, value);
    }

    boolean get(Player ply, StateFlag flag)
    {
        if (!Flags.containsKey(ply.getUniqueId()))
            Flags.put(ply.getUniqueId(), new HashMap<>());
        if (!Flags.get(ply.getUniqueId()).containsKey(flag))
            Flags.get(ply.getUniqueId()).put(flag, false);
        return Flags.get(ply.getUniqueId()).get(flag);
    }
}

enum StateFlag
{
    superLeap, slamming, inAir
}
