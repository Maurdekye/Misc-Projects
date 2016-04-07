package com.ncolaprete.lootcrate;

import com.mysql.jdbc.Util;
import net.minecraft.server.v1_9_R1.ChatComponentScore;
import net.minecraft.server.v1_9_R1.SystemUtils;
import net.minecraft.server.v1_9_R1.TileEntityChest;
import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.block.Chest;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.ConsoleCommandSender;
import org.bukkit.configuration.ConfigurationSection;
import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.craftbukkit.v1_9_R1.block.CraftChest;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.*;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.ProjectileLaunchEvent;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.FireworkMeta;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.inventory.meta.PotionMeta;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;
import org.bukkit.util.BlockIterator;
import org.bukkit.util.Vector;

import javax.lang.model.type.ArrayType;
import java.lang.reflect.Field;
import java.util.*;

public class LootCrate extends JavaPlugin implements Listener, CommandExecutor{

    private ArrayList<Crate> allCrates;
    private ArrayList<CrateLayout> crateLayouts;
    HashMap<UUID, Long> tempCreativeTimestamps;

    private CustomConfig cratePositionConfig;
    private CustomConfig crateLayoutConfig;
    CustomConfig tempCreativeTrackerConfig;

    private ConsoleCommandSender csend;

    // Overridden Methods

    public void onEnable()
    {
        getServer().getPluginManager().registerEvents(this, this);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::checkForSpecialItems, 0, 20);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::checkForCreativeTimeUp, 0, 1200);

        csend = getServer().getConsoleSender();

        allCrates = new ArrayList<>();
        crateLayouts = new ArrayList<>();
        tempCreativeTimestamps = new HashMap<>();

        // load up configs
        cratePositionConfig = new CustomConfig(this, "crate_positions.yml");
        crateLayoutConfig = new CustomConfig(this, "crate_layouts.yml");
        tempCreativeTrackerConfig = new CustomConfig(this, "temp_creatives.yml");
        crateLayoutConfig.getConfig().options().copyDefaults(true);
        cratePositionConfig.saveConfig();
        crateLayoutConfig.saveConfig();
        tempCreativeTrackerConfig.saveConfig();

        // load in crate layouts
        for (String key : crateLayoutConfig.getConfig().getKeys(false))
        {
            String type = key;
            String printname = crateLayoutConfig.getConfig().getString(key + ".name");
            printname = ChatColor.translateAlternateColorCodes('?', printname);
            String reqKeyName = crateLayoutConfig.getConfig().getString(key + ".required_key");
            Prize reqKey;
            try
            {
                reqKey = Prize.valueOf(reqKeyName.toUpperCase());
            } catch (Exception e)
            {
                csend.sendMessage(ChatColor.RED + "Error! Unknown prize: " + reqKeyName);
                continue;
            }
            ArrayList<Reward> rewardList = new ArrayList<>();
            ConfigurationSection rewardsSection = crateLayoutConfig.getConfig().getConfigurationSection(key + ".rewards");
            for (String rewardKey : rewardsSection.getKeys(false))
            {
                String prizeName = rewardsSection.getString(rewardKey + ".prize");
                Prize prize;
                double rewardChance;
                int amount;
                try {
                    prize = Prize.valueOf(prizeName.toUpperCase());
                } catch (Exception e) {
                    csend.sendMessage(ChatColor.RED + "Error! Unknown prize: " + prizeName);
                    continue;
                }
                try {
                    rewardChance = Double.parseDouble(rewardsSection.getString(rewardKey + ".chance"));
                } catch (Exception e) {
                    csend.sendMessage(ChatColor.RED + "Error! '" + rewardsSection.getString(rewardKey + ".chance") + "' is not a number!");
                    continue;
                }
                try {
                    amount = Integer.parseInt(rewardsSection.getString(rewardKey + ".amount"));
                } catch (Exception e) {
                    csend.sendMessage(ChatColor.RED + "Error! '" + rewardsSection.getString(rewardKey + ".amount") + "' is not an integer!");
                    continue;
                }

                rewardList.add(new Reward(prize, amount, rewardChance));
            }
            crateLayouts.add(new CrateLayout(printname, type, reqKey, rewardList));
        }

        if (crateLayouts.size() == 0)
        {
            csend.sendMessage(ChatColor.RED + "Critical Error; No crate layouts detected! Disabling plugin.");
            getPluginLoader().disablePlugin(this);
        }

        // load in crate locations
        for (String key : cratePositionConfig.getConfig().getKeys(false))
        {
            Location pos = Utility.deserializeLocation(getServer(), key);
            String layoutname = cratePositionConfig.getConfig().getString(key);
            CrateLayout layout = null;
            for (CrateLayout l : crateLayouts)
            {
                if (l.type.equalsIgnoreCase(layoutname))
                {
                    layout = l;
                    break;
                }
            }
            if (layout == null)
                continue;
            addCrate(new Crate(pos.getBlock(), layout));
        }

        // load in temporary creatives
        for (String key : tempCreativeTrackerConfig.getConfig().getKeys(false))
        {
            tempCreativeTimestamps.put(UUID.fromString(key), tempCreativeTrackerConfig.getConfig().getLong(key));
        }
    }

    public void onDisable()
    {
        cratePositionConfig.saveConfig();
        tempCreativeTrackerConfig.saveConfig();
    }

    private void checkForSpecialItems()
    {
        for (Player ply : getServer().getOnlinePlayers())
        {
            // Frostspark Cleats
            if (Utility.itemHasLoreLine(ply.getInventory().getBoots(), ChatColor.BLACK + "frostspark_cleats"))
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.SPEED, 40, 2), true);
                ply.addPotionEffect(new PotionEffect(PotionEffectType.JUMP, 40, 1), true);
            }

            // Lucky Trousers
            if (Utility.itemHasLoreLine(ply.getInventory().getLeggings(), ChatColor.BLACK + "lucky_trousers"))
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.LUCK, 40, 2), true);
            }

            // Knackerbreaker Chesterplate
            if (Utility.itemHasLoreLine(ply.getInventory().getChestplate(), ChatColor.BLACK + "knackerbreaker_chesterplate"))
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.ABSORPTION, 40, 0), true);
            }

            // Hydrodyne Helmet
            if (Utility.itemHasLoreLine(ply.getInventory().getHelmet(), ChatColor.BLACK + "hydrodyne_helmet"))
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.NIGHT_VISION, 250, 0), true);
                ply.addPotionEffect(new PotionEffect(PotionEffectType.WATER_BREATHING, 40, 0), true);
            }

            // Giga Drill Breaker
            if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), ChatColor.BLACK + "giga_drill_breaker"))
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.FAST_DIGGING, 25, 3), true);
            }

            // Unyielding Battersea
            if ((Utility.itemHasLoreLine(ply.getInventory().getItemInOffHand(), ChatColor.BLACK + "unyielding_battersea") ||
                    Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), ChatColor.BLACK + "unyielding_battersea")) &&
                    ply.isBlocking())
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.DAMAGE_RESISTANCE, 40, 0), true);
                ply.addPotionEffect(new PotionEffect(PotionEffectType.FIRE_RESISTANCE, 40, 0), true);
            }

            // Veilstrike Bow
            if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), ChatColor.BLACK + "veilstrike_bow"))
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.INVISIBILITY, 40, 0), true);
            }

            // Heaven's Blade
            if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), ChatColor.BLACK + "heavens_blade"))
            {
                ply.addPotionEffect(new PotionEffect(PotionEffectType.INCREASE_DAMAGE, 40, 4), true);
            }
        }
    }

    private void checkForCreativeTimeUp()
    {
        for (UUID plyId : tempCreativeTimestamps.keySet())
        {
            Player ply = getServer().getPlayer(plyId);
            if (ply == null)
                continue;
            if (System.currentTimeMillis() > tempCreativeTimestamps.get(plyId))
            {
                ply.setGameMode(GameMode.SURVIVAL);
                ply.sendMessage(ChatColor.DARK_AQUA + "Your creative time is up.");
                tempCreativeTimestamps.remove(ply.getUniqueId());
                tempCreativeTrackerConfig.getConfig().set(plyId.toString(), null);
                tempCreativeTrackerConfig.saveConfig();
            }
        }
    }

    public boolean onCommand(CommandSender sender, Command command, String label, String[] args)
    {
        Player ply = sender instanceof Player ? (Player) sender : null;

        // givecrate
        if (command.getName().equalsIgnoreCase("givecrate"))
        {

            // Find chest layout to use
            CrateLayout layout = null;
            if (args.length == 0)
            {
                layout = crateLayouts.get((int)(Math.random() * crateLayouts.size()));
            }
            else
            {
                String type = args[0];
                for (CrateLayout l : crateLayouts)
                {
                    if (l.type.equalsIgnoreCase(type))
                    {
                        layout = l;
                        break;
                    }
                }
                if (layout == null)
                {
                    sender.sendMessage(ChatColor.RED + "No crate layout of type " + type);
                    return true;
                }
            }

            // Find player to give crate to
            Player target = ply;
            if (args.length >= 2)
            {
                String targetname = "";
                for (int i=1;i<args.length;i++)
                    targetname += args[i] + " ";
                targetname = targetname.trim().toLowerCase();
                Player newtarget = getServer().getPlayer(targetname);
                if (newtarget == null)
                {
                    sender.sendMessage(ChatColor.RED + "Could not find player with the name '" + targetname + "'");
                    return true;
                }
                target = newtarget;
            }

            // Check if sender is player
            if (target == null)
            {
                sender.sendMessage(ChatColor.RED + "You must be a player to use this command on yourself");
                return true;
            }
            else
                target = ply;

            // give crate
            target.getInventory().addItem(getCrateItemstack(layout));
        }

        // spawncrate
        else if (command.getName().equalsIgnoreCase("spawncrate"))
        {

            // Check if sender is player
            if (ply == null)
            {
                sender.sendMessage(ChatColor.RED + "You must be a player to use this command");
                return true;
            }

            // Find chest layout to use
            CrateLayout layout = null;
            if (args.length == 0)
            {
                layout = crateLayouts.get((int)(Math.random() * crateLayouts.size()));
            }
            else
            {
                String type = args[0];
                for (CrateLayout l : crateLayouts)
                {
                    if (l.type.equalsIgnoreCase(type))
                    {
                        layout = l;
                        break;
                    }
                }
                if (layout == null)
                {
                    sender.sendMessage(ChatColor.RED + "No crate layout of type " + type);
                    return true;
                }
            }

            // Find block position to use
            Block location = null;
            BlockIterator biter = new BlockIterator(ply, 40);
            while (biter.hasNext())
            {
                Block next = biter.next();
                if (next.getType() != Material.AIR)
                    break;
                location = next;
            }
            if (location == null)
            {
                sender.sendMessage(ChatColor.RED + "Unable to place chest.");
                return true;
            }

            // Place crate
            addCrate(new Crate(location, layout));
            sender.sendMessage(layout.printname + ChatColor.AQUA + " Placed");
        }

        // givereward
        else if (command.getName().equalsIgnoreCase("givereward"))
        {

            // Check if sender is player
            if (ply == null)
            {
                sender.sendMessage(ChatColor.RED + "You must be a player to use this command.");
                return true;
            }

            // Check if they've given at least one argument
            if (args.length == 0)
            {
                String rewardslist = "Available rewards are: ";
                for (Prize p : Prize.values())
                    rewardslist += p.name().toLowerCase() + ", ";
                ply.sendMessage(rewardslist.substring(0, rewardslist.length()-2));
                return false;
            }

            // Find prize type to give
            Prize type = null;
            for (Prize p : Prize.values())
            {
                if (p.toString().equalsIgnoreCase(args[0]))
                {
                    type = p;
                    break;
                }
            }
            if (type == null)
            {
                sender.sendMessage(ChatColor.RED + "Could not find prize " + args[0] + ".");
                return true;
            }

            // Find amount to give
            int amount = 1;
            if (args.length >= 2)
            {
                try
                {
                    amount = Integer.parseInt(args[1]);
                }
                catch (Exception e)
                {
                    sender.sendMessage(ChatColor.RED + "'" + args[1] + "' is not a number.");
                    return true;
                }
            }

            // find block to give from
            Block chestBlock = null;
            BlockIterator biter = new BlockIterator(ply, 40);
            while (biter.hasNext())
            {
                chestBlock = biter.next();
                if (chestBlock.getType() == Material.CHEST)
                    break;
            }
            if (chestBlock.getType() != Material.CHEST)
            {
                sender.sendMessage(ChatColor.RED + "Unable to spawn prize; please look at a chest.");
                return true;
            }

            // give prize
            type.giveReward(this, ply, amount, chestBlock);
        }
        return true;
    }

    // Event Handlers

    @EventHandler
    public void playerInteract(PlayerInteractEvent ev)
    {
        if (ev.getAction() == Action.RIGHT_CLICK_BLOCK && !ev.getPlayer().isSneaking())
        {
            Block block = ev.getClickedBlock();
            Crate crateToOpen = null;
            for (Crate crate : allCrates)
            {
                if (crate.location.equals(block))
                {
                    crateToOpen = crate;
                    break;
                }
            }
            if (crateToOpen == null)
                return;
            if (!ev.getPlayer().hasPermission("lootcrate.opencrate"))
            {
                ev.getPlayer().sendMessage(ChatColor.RED + "You don't have permission to open crates.");
                ev.setCancelled(true);
                return;
            }
            ItemStack handItem = ev.getPlayer().getInventory().getItemInMainHand();
            if (crateToOpen.isKeyValid(handItem))
            {
                if (handItem.getAmount() > 1)
                    handItem.setAmount(handItem.getAmount() - 1);
                else
                    ev.getPlayer().getInventory().remove(handItem);
                crateToOpen.unlockAndGivePrize(this, ev.getPlayer());
                removeCrate(crateToOpen);
            }
            else
            {
                ev.setCancelled(true);
                ev.getPlayer().openInventory(crateToOpen.showContents(this, ev.getPlayer()));
            }
        }
    }

    @EventHandler
    public void blockBreak(BlockBreakEvent ev)
    {
        for (int i=0;i<allCrates.size();i++)
        {
            if (allCrates.get(i).location.equals(ev.getBlock()))
            {
                ev.setCancelled(true);
                ItemStack crateDrop = getCrateItemstack(allCrates.get(i).layout);
                ev.getBlock().setType(Material.AIR);
                ev.getBlock().getWorld().dropItemNaturally(ev.getBlock().getLocation().add(0.5, 0.5, 0.5), crateDrop);
                removeCrate(i);
                break;
            }
        }
    }

    @EventHandler
    public void blockPlace(BlockPlaceEvent ev)
    {
        CrateLayout newCrate = null;
        BlockFace[] cardinalDirections = new BlockFace[] {BlockFace.NORTH, BlockFace.SOUTH, BlockFace.EAST, BlockFace.WEST};
        for (CrateLayout l : crateLayouts)
        {
            if (Utility.itemHasLoreLine(ev.getItemInHand(), ChatColor.BLACK + l.type))
            {
                newCrate = l;
                break;
            }
        }
        if (newCrate != null)
        {
            for (BlockFace f : cardinalDirections)
            {
                if (ev.getBlock().getRelative(f).getType() == Material.CHEST)
                {
                    ev.setCancelled(true);
                    return;
                }
            }
        }
        else if (ev.getItemInHand().getType() == Material.CHEST)
        {
            for (BlockFace f : cardinalDirections)
            {
                for (Crate c : allCrates)
                {
                    if (ev.getBlock().getRelative(f).equals(c.location))
                    {
                        ev.setCancelled(true);
                        return;
                    }
                }
            }
        }
        if (newCrate == null)
            return;
        addCrate(new Crate(ev.getBlock(), newCrate));
    }

    @EventHandler
    public void inventoryClick(InventoryClickEvent ev)
    {
        for (CrateLayout l : crateLayouts)
        {
            if ((l.getPrintname(true)).equals(ev.getInventory().getName()))
            {
                ev.setCancelled(true);
                break;
            }
        }
    }

    @EventHandler
    public void projectileLaunch(ProjectileLaunchEvent ev)
    {
        if (ev.getEntity() instanceof Arrow && ev.getEntity().getShooter() instanceof Player)
        {
            Player ply = (Player) ev.getEntity().getShooter();
            if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), ChatColor.BLACK + "veilstrike_bow"))
            {
                ev.getEntity().setVelocity(ev.getEntity().getVelocity().multiply(4));
            }
        }
    }

    private void removeCrate(Crate c)
    {
        cratePositionConfig.getConfig().set(Utility.serializeLocation(c.location.getLocation()), null);
        cratePositionConfig.saveConfig();
        allCrates.remove(c);
    }

    private void removeCrate(int index)
    {
        cratePositionConfig.getConfig().set(Utility.serializeLocation(allCrates.get(index).location.getLocation()), null);
        cratePositionConfig.saveConfig();
        allCrates.remove(index);
    }

    private void addCrate(Crate c)
    {
        allCrates.add(c);
        cratePositionConfig.getConfig().set(Utility.serializeLocation(c.location.getLocation()), c.layout.type);
        cratePositionConfig.saveConfig();
    }

    private ItemStack getCrateItemstack(CrateLayout l)
    {
        ItemStack crateDrop = Utility.setName(new ItemStack(Material.CHEST), l.getPrintname(true));
        crateDrop = Utility.addLoreLine(crateDrop, ChatColor.BLACK + l.type);
        return crateDrop;
    }

}

class Utility
{
    static boolean itemHasLoreLine(ItemStack item, String line)
    {
        if (item == null)
            return false;
        ItemMeta meta = item.getItemMeta();
        if (meta == null)
            return false;
        List<String> lore = meta.getLore();
        if (lore == null)
            return false;
        return lore.contains(line);
    }
    static ItemStack addLoreLine(ItemStack item, String line)
    {
        ItemMeta meta = item.getItemMeta();
        if (meta == null)
        {
            meta.setLore(Arrays.asList(line));
        }
        else
        {
            List<String> lore = meta.getLore();
            if (lore == null)
                lore = new ArrayList<>();
            lore.add(line);
            meta.setLore(lore);
        }
        item.setItemMeta(meta);
        return item;
    }

    static ItemStack setName(ItemStack item, String name)
    {
        ItemMeta meta = item.getItemMeta();
        meta.setDisplayName(name);
        item.setItemMeta(meta);
        return item;
    }

    public static String getName(ItemStack item)
    {
        ItemMeta meta = item.getItemMeta();
        if (meta == null)
            return "";
        if (meta.getDisplayName() == null)
            return "";
        return meta.getDisplayName();
    }

    static int randomInt(int start, int end)
    {
        return (int)(Math.random() * (end - start)) + start;
    }

    static void setChestInventoryName(Block chestblock, String name)
    {
        CraftChest chest = (CraftChest) chestblock.getState();
        try
        {
            Field inventoryField = chest.getClass().getDeclaredField("chest");
            inventoryField.setAccessible(true);
            TileEntityChest teChest = (TileEntityChest) inventoryField.get(chest);
            teChest.a(name);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
    }

    static String serializeLocation(Location loc)
    {
        StringBuilder sb = new StringBuilder();
        sb.append(loc.getWorld().getName());
        sb.append("?");
        sb.append(loc.getBlockX());
        sb.append("?");
        sb.append(loc.getBlockY());
        sb.append("?");
        sb.append(loc.getBlockZ());
        return sb.toString();
    }

    static Location deserializeLocation(Server server, String serial)
    {
        server.getConsoleSender().sendMessage("deserializing " + serial);
        String[] parts = serial.split("\\?");
        World world = server.getWorld(parts[0]);
        int x = Integer.parseInt(parts[1]);
        int y = Integer.parseInt(parts[2]);
        int z = Integer.parseInt(parts[3]);
        return world.getBlockAt(x, y, z).getLocation();
    }
}

class Crate
{
    public Block location;
    public CrateLayout layout;

    public Crate(Block location, CrateLayout layout)
    {
        this.location = location;
        this.layout = layout;

        if (this.location.getType() != Material.CHEST)
            this.location.setType(Material.CHEST);
    }

    public void unlockAndGivePrize(LootCrate plugin, Player rewardee)
    {
        layout.givePrize(plugin, rewardee, location);
        Utility.setChestInventoryName(location, layout.getPrintname(false));
    }

    public Inventory showContents(LootCrate plugin, Player ply)
    {
        return layout.showContents(plugin, ply, location);
    }

    public boolean isKeyValid(ItemStack key)
    {
        return layout.isKeyValid(key);
    }
}

class CrateLayout
{
    private static final String LockedTag = ChatColor.RED + " [Locked]";
    private static final String UnlockedTag = ChatColor.YELLOW + " [Unlocked]";

    String printname;
    String type;
    private Prize keyRequired;
    private ArrayList<Reward> contents;

    CrateLayout(String printname, String type, Prize keyRequired, ArrayList<Reward> contents)
    {
        this.printname = printname;
        this.type = type;
        this.keyRequired = keyRequired;
        this.contents = contents;
    }

    void givePrize(LootCrate plugin, Player rewardee, Block location)
    {
        double sum = 0;
        for (Reward r : contents)
            sum += r.rewardChance;
        double rand = Math.random() * sum;
        int prizeIndex = contents.size() - 1;
        for (int i=0;i<contents.size();i++)
        {
            if (rand < contents.get(i).rewardChance)
            {
                prizeIndex = i;
                break;
            }
            rand -= contents.get(i).rewardChance;
        }
        Reward chosen = contents.get(prizeIndex);
        chosen.item.giveReward(plugin, rewardee, chosen.amount, location);
        if (chosen.item == Prize.NOTHING)
            rewardee.playSound(rewardee.getLocation(), Sound.BLOCK_NOTE_SNARE, 1, 1);
        else
            rewardee.playSound(rewardee.getLocation(), Sound.ENTITY_PLAYER_LEVELUP, 1, 1);
    }

    Inventory showContents(LootCrate plugin, Player ply, Block chestblock)
    {
        Inventory display = Bukkit.createInventory(null, contents.size(), getPrintname(true));
        for (Reward r : contents)
        {
            ItemStack displayItem =  r.item.getVisualisation(plugin, ply, r.amount, chestblock);
            displayItem = Utility.addLoreLine(displayItem, ChatColor.WHITE + "%" + r.rewardChance + " chance");
            display.addItem(displayItem);
        }
        return display;
    }

    boolean isKeyValid(ItemStack key)
    {
        return Utility.itemHasLoreLine(key, keyRequired.getLoreTag());
    }

    String getPrintname(boolean isLocked)
    {
        if (isLocked)
            return printname + LockedTag;
        return printname + UnlockedTag;
    }
}

class Reward
{
    Prize item;
    int amount;
    double rewardChance;

    Reward(Prize item, int amount, double rewardChance)
    {
        this.item = item;
        this.amount = amount;
        this.rewardChance = rewardChance;
    }
}

enum Prize
{

    // keys

    ANCIENT_KEY (params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.DARK_GRAY + "Ancient Key");
        keyMeta.setLore(Arrays.asList("An old key from long ago. History has long forgotten what it unlocks.", ChatColor.BLACK + "ancient_key"));
        key.setItemMeta(keyMeta);
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Ancient Keys!");
        else
            params.rewardee.sendMessage("You got an Ancient Key!");
        return Collections.singletonList(key);
    }, params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.DARK_GRAY + "Ancient Key");
        keyMeta.setLore(Collections.singletonList("An old key from long ago. History has long forgotten what it unlocks."));
        key.setItemMeta(keyMeta);
        return key;
    }),

    LEGENDARY_KEY (params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.YELLOW + "" + ChatColor.BOLD + "Legendary Key");
        keyMeta.setLore(Arrays.asList("An incredibly rare key of legends. It unlocks untold riches.", ChatColor.BLACK + "legendary_key"));
        key.setItemMeta(keyMeta);
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Legendary Keys!");
        else
            params.rewardee.sendMessage("You got a Legendary Key!");
        return Collections.singletonList(key);
    }, params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.LIGHT_PURPLE + "Legendary Key");
        keyMeta.setLore(Collections.singletonList("An incredibly rare key of legends. It unlocks untold riches."));
        key.setItemMeta(keyMeta);
        return key;
    }),

    MYSTICAL_KEY (params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.LIGHT_PURPLE + "" + ChatColor.ITALIC + "Mystical Key");
        keyMeta.setLore(Arrays.asList("A Mysticalal key of unknown origin. Surely very rare.", ChatColor.BLACK + "Mystical_key"));
        key.setItemMeta(keyMeta);
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Mystical Keys!");
        else
            params.rewardee.sendMessage("You got a Mystical Key!");
        return Collections.singletonList(key);
    }, params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.LIGHT_PURPLE + "" + ChatColor.ITALIC + "Mystical Key");
        keyMeta.setLore(Collections.singletonList("A Mysticalal key of unknown origin. Surely very rare."));
        key.setItemMeta(keyMeta);
        return key;
    }),

    COMMON_KEY (params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.GREEN + "Common Key");
        keyMeta.setLore(Arrays.asList("A normal crate key.", ChatColor.BLACK + "common_key"));
        key.setItemMeta(keyMeta);
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Common Keys!");
        else
            params.rewardee.sendMessage("You got a Common Key!");
        return Collections.singletonList(key);
    }, params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.GREEN + "Common Key");
        keyMeta.setLore(Collections.singletonList("A normal crate key."));
        key.setItemMeta(keyMeta);
        return key;
    }),

    // reward items

    ULTIMATE_REWARD (params -> {
        params.rewardee.setGameMode(GameMode.CREATIVE);
        params.plugin.tempCreativeTrackerConfig.getConfig().set(params.rewardee.getUniqueId().toString(), System.currentTimeMillis() + params.amountToGive * 3600000);
        params.plugin.tempCreativeTimestamps.put(params.rewardee.getUniqueId(), System.currentTimeMillis() + params.amountToGive * 3600000);
        params.plugin.tempCreativeTrackerConfig.saveConfig();
        if (params.amountToGive == 1)
            params.rewardee.sendMessage("You have received the ultimate reward: " + ChatColor.BOLD + "you may be in creative for 1 hour.");
        else
            params.rewardee.sendMessage("You have received the ultimate reward: " + ChatColor.BOLD + "you may be in creative for " + params.amountToGive + " hours.");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.COMMAND);
        return Utility.setName(item, ChatColor.UNDERLINE + "" + ChatColor.BOLD + "" + "The Ultimate Reward");
    }),

    FROSTSPARK_CLEATS (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_BOOTS);
        item = Utility.setName(item, ChatColor.YELLOW + "Frostspark Cleats");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The cleats grant improved mobility");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "frostspark_cleats");
        item.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 1);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.FROST_WALKER, 2);
        item.addEnchantment(Enchantment.PROTECTION_FALL, 4);
        params.rewardee.sendMessage(ChatColor.YELLOW + "You got the Frostspark Cleats!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_BOOTS);
        return Utility.setName(item, ChatColor.YELLOW + "Frostspark Cleats");
    }),

    WATERGLIDE_BOOTS (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_BOOTS);
        item = Utility.setName(item, ChatColor.AQUA + "Waterglide Boots");
        item.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 1);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.DEPTH_STRIDER, 3);
        params.rewardee.sendMessage(ChatColor.AQUA + "You got the Waterglide Boots!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_BOOTS);
        return Utility.setName(item, ChatColor.AQUA + "Waterglide Boots");
    }),

    LUCKY_TROUSERS (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_LEGGINGS);
        item = Utility.setName(item, ChatColor.GREEN + "Lucky Trousers");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The trousers grant increased luck");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "lucky_trousers");
        item.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 1);
        item.addEnchantment(Enchantment.DURABILITY, 1);
        params.rewardee.sendMessage(ChatColor.GREEN + "You got the Lucky Trousers!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_LEGGINGS);
        return Utility.setName(item, ChatColor.GREEN + "Lucky Trousers");
    }),

    KNACKERBREAKER_CHESTERPLATE (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_CHESTPLATE);
        item = Utility.setName(item, ChatColor.GOLD + "Knackerbreaker Chesterplate");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The chestplate grants increased health absorption");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "knackerbreaker_chesterplate");
        item.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 2);
        item.addEnchantment(Enchantment.PROTECTION_EXPLOSIONS, 1);
        item.addEnchantment(Enchantment.PROTECTION_FIRE, 1);
        item.addEnchantment(Enchantment.PROTECTION_PROJECTILE, 1);
        item.addEnchantment(Enchantment.THORNS, 3);
        item.addEnchantment(Enchantment.DURABILITY, 1);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "You got the Knackerbreaker Chesterplate!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_CHESTPLATE);
        return Utility.setName(item, ChatColor.GOLD + "Knackerbreaker Chesterplate");
    }),

    HYDRODYNE_HELMET (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_HELMET);
        item = Utility.setName(item, ChatColor.BLUE + "Hydrodyne Helmet");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The helmet grants improved underwater and visual acuity");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "hydrodyne_helmet");
        item.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 1);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.OXYGEN, 3);
        item.addEnchantment(Enchantment.WATER_WORKER, 1);
        params.rewardee.sendMessage(ChatColor.BLUE + "You got the Hydrodyne Helmet!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_HELMET);
        return Utility.setName(item, ChatColor.BLUE + "Hydrodyne Helmet");
    }),

    GIGA_DRILL_BREAKER (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_PICKAXE);
        item = Utility.setName(item, ChatColor.AQUA + "" + ChatColor.BOLD + "Giga Drill Breaker");
        item = Utility.addLoreLine(item, ChatColor.AQUA + "Bust through the heavens with your Drill!");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "giga_drill_breaker");
        item.addEnchantment(Enchantment.DIG_SPEED, 5);
        item.addEnchantment(Enchantment.DURABILITY, 3);
        params.rewardee.sendMessage(ChatColor.AQUA + "You got the Giga Drill Breaker; thrust through the heavens with your spirit!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_PICKAXE);
        return Utility.setName(item, ChatColor.AQUA + "" + ChatColor.BOLD + "Giga Drill Breaker");
    }),

    UNYIELDING_BATTERSEA (params -> {
        ItemStack item = new ItemStack(Material.SHIELD);
        item = Utility.setName(item, ChatColor.YELLOW + "Unyielding Battersea");
        item = Utility.addLoreLine(item, "An olden shield used by unending legions");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The battersea grants increase resistances while equipped");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "unyielding_battersea");
        item.addEnchantment(Enchantment.DURABILITY, 3);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "You got the Unyielding Battersea!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.SHIELD);
        return Utility.setName(item, ChatColor.YELLOW + "Unyielding Battersea");
    }),

    VEILSTRIKE_BOW (params -> {
        ItemStack item = new ItemStack(Material.BOW);
        item = Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.ITALIC + "Veilstrike Bow");
        item = Utility.addLoreLine(item, "An ancient, powerful bow used by an extremely skilled marksman");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The bow grants invisibility and immense arrow speed");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "veilstrike_bow");
        item.addEnchantment(Enchantment.ARROW_DAMAGE, 5);
        item.addEnchantment(Enchantment.ARROW_KNOCKBACK, 2);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "" + ChatColor.ITALIC + "You got the Veilstrike Bow!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.BOW);
        return Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.ITALIC + "Veilstrike Bow");
    }),

    HEAVENS_BLADE (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_SWORD);
        item = Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.BOLD + "Heaven's Blade");
        item = Utility.addLoreLine(item, "A godly blade that weilds incredible desctructive power");
        item = Utility.addLoreLine(item, ChatColor.BLACK + "heavens_blade");
        item.addEnchantment(Enchantment.DAMAGE_ALL, 5);
        item.addEnchantment(Enchantment.KNOCKBACK, 2);
        item.addEnchantment(Enchantment.FIRE_ASPECT, 2);
        item.addEnchantment(Enchantment.DURABILITY, 3);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "" + ChatColor.BOLD + "You got Heaven's Blade!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_SWORD);
        return Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.BOLD + "Heaven's Blade");
    }),

    IRON_COMBAT_SET (params -> {
        ArrayList<ItemStack> rewards = new ArrayList<>();
        rewards.add(new ItemStack(Material.IRON_SWORD));
        rewards.add(new ItemStack(Material.IRON_AXE));
        rewards.add(new ItemStack(Material.SHIELD));
        rewards.add(new ItemStack(Material.BOW));
        rewards.add(new ItemStack(Material.ARROW, params.amountToGive));
        rewards.add(new ItemStack(Material.IRON_HELMET));
        rewards.add(new ItemStack(Material.IRON_CHESTPLATE));
        rewards.add(new ItemStack(Material.IRON_LEGGINGS));
        rewards.add(new ItemStack(Material.IRON_BOOTS));
        params.rewardee.sendMessage("You got full iron combat gear!");
        return rewards;
    }, params -> {
        ItemStack item = new ItemStack(Material.IRON_CHESTPLATE);
        return Utility.setName(item, ChatColor.AQUA + "Full Iron Combat Gear");
    }),

    IRON_TOOLSET (params -> {
        ArrayList<ItemStack> rewards = new ArrayList<>();
        rewards.add(new ItemStack(Material.IRON_PICKAXE));
        rewards.add(new ItemStack(Material.IRON_AXE));
        rewards.add(new ItemStack(Material.IRON_SWORD));
        rewards.add(new ItemStack(Material.IRON_SPADE));
        rewards.add(new ItemStack(Material.IRON_HOE));
        params.rewardee.sendMessage("You got a full iron toolset!");
        return rewards;
    }, params -> {
        ItemStack item = new ItemStack(Material.IRON_PICKAXE);
        return Utility.setName(item, ChatColor.AQUA + "Full Iron Toolset");
    }),

    DIAMONDS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Diamonds!");
        else
            params.rewardee.sendMessage("You got a Diamond!");
        return Collections.singletonList(new ItemStack(Material.DIAMOND, params.amountToGive));
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND, 1);
        if (params.amountToGive > 1)
            item = Utility.setName(item, ChatColor.DARK_AQUA + "" + params.amountToGive + " Diamonds");
        else
            item = Utility.setName(item, ChatColor.DARK_AQUA + "1 Diamond");
        return item;
    }),

    IRON_BARS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " iron ingots!");
        else
            params.rewardee.sendMessage("You got an iron ingot!");
        return Collections.singletonList(new ItemStack(Material.IRON_INGOT, params.amountToGive));
    }, params -> {
        ItemStack item = new ItemStack(Material.IRON_INGOT, 1);
        if (params.amountToGive > 1)
            item = Utility.setName(item, ChatColor.DARK_AQUA + "" + params.amountToGive + " Iron Ingots");
        else
            item = Utility.setName(item, ChatColor.DARK_AQUA + "1 Iron Ingot");
        return item;
    }),

    GOLD_BARS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " gold ingots!");
        else
            params.rewardee.sendMessage("You got a gold ingot!");
        return Collections.singletonList(new ItemStack(Material.GOLD_INGOT, params.amountToGive));
    }, params -> {
        ItemStack item = new ItemStack(Material.IRON_INGOT, 1);
        if (params.amountToGive > 1)
            item = Utility.setName(item, ChatColor.DARK_AQUA + "" + params.amountToGive + " Gold Ingots");
        else
            item = Utility.setName(item, ChatColor.DARK_AQUA + "1 Gold Ingot");
        return item;
    }),

    RAW_ORE_BLOCKS (params -> {
        ArrayList<ItemStack> rewards = new ArrayList<>();
        HashMap<Material, Integer> multipliers = new HashMap<>();
        multipliers.put(Material.COAL_ORE, 6);
        multipliers.put(Material.IRON_ORE, 4);
        multipliers.put(Material.GOLD_ORE, 3);
        multipliers.put(Material.REDSTONE_ORE, 5);
        multipliers.put(Material.LAPIS_ORE, 5);
        multipliers.put(Material.DIAMOND_ORE, 1);
        multipliers.put(Material.EMERALD_ORE, 1);
        for (Material key : multipliers.keySet())
        {
            int basecount = params.amountToGive * multipliers.get(key);
            ItemStack item = new ItemStack(key, Utility.randomInt(basecount, basecount*2));
            rewards.add(item);
        }
        params.rewardee.sendMessage(ChatColor.AQUA + "You got various raw ore blocks!");
        return rewards;
    }, params -> {
        ItemStack item = new ItemStack(Material.REDSTONE_ORE);
        return Utility.setName(item, ChatColor.AQUA + "Assorted Raw Ore Blocks");
    }),

    ASSORTED_ORES (params -> {
        ArrayList<ItemStack> rewards = new ArrayList<>();
        HashMap<Material, Integer> multipliers = new HashMap<>();
        multipliers.put(Material.COBBLESTONE, 24);
        multipliers.put(Material.COAL, 12);
        multipliers.put(Material.IRON_INGOT, 8);
        multipliers.put(Material.GOLD_INGOT, 6);
        multipliers.put(Material.REDSTONE, 18);
        multipliers.put(Material.INK_SACK, 15);
        multipliers.put(Material.DIAMOND, 2);
        multipliers.put(Material.EMERALD, 3);
        for (Material key : multipliers.keySet())
        {
            int basecount = params.amountToGive * multipliers.get(key);
            ItemStack item = new ItemStack(key, Utility.randomInt(basecount, basecount*2));
            if (key == Material.INK_SACK)
                item.setDurability((short)4);
            rewards.add(item);
        }
        params.rewardee.sendMessage(ChatColor.AQUA + "You got an assortment of ores!");
        return rewards;
    }, params -> {
        ItemStack item = new ItemStack(Material.COAL);
        return Utility.setName(item, ChatColor.AQUA + "Assorted Ores");
    }),

    ASSORTMENT (params -> {
        ArrayList<ItemStack> rewards = new ArrayList<>();
        HashMap<Material, Integer> multipliers = new HashMap<>();
        multipliers.put(Material.GOLD_NUGGET, 6);
        multipliers.put(Material.REDSTONE, 16);
        multipliers.put(Material.INK_SACK, 9);
        multipliers.put(Material.ARROW, 12);
        multipliers.put(Material.SULPHUR, 8);
        multipliers.put(Material.BLAZE_POWDER, 6);
        multipliers.put(Material.APPLE, 3);
        multipliers.put(Material.LOG, 5);
        for (Material key : multipliers.keySet())
        {
            int basecount = params.amountToGive * multipliers.get(key);
            ItemStack item = new ItemStack(key, Utility.randomInt(basecount, basecount*2));
            if (key == Material.INK_SACK)
                item.setDurability((short)4);
            rewards.add(item);
        }
        params.rewardee.sendMessage(ChatColor.GREEN + "You got a random assortment of items.");
        return rewards;
    }, params -> {
        ItemStack item = new ItemStack(Material.CHEST);
        item = Utility.setName(item, ChatColor.GREEN + "An Assortment of Items");
        return item;
    }),

    PLANTS (params -> {
        ArrayList<ItemStack> rewards = new ArrayList<>();
        HashMap<Material, Integer> multipliers = new HashMap<>();
        multipliers.put(Material.LOG, 16);
        multipliers.put(Material.SAPLING, 6);
        multipliers.put(Material.APPLE, 12);
        multipliers.put(Material.YELLOW_FLOWER, 4);
        multipliers.put(Material.SEEDS, 24);
        multipliers.put(Material.MELON_SEEDS, 4);
        multipliers.put(Material.PUMPKIN_SEEDS, 3);
        multipliers.put(Material.POTATO, 7);
        multipliers.put(Material.CARROT, 4);
        for (Material key : multipliers.keySet())
        {
            int basecount = params.amountToGive * multipliers.get(key);
            ItemStack item = new ItemStack(key, Utility.randomInt(basecount, basecount*2));
            rewards.add(item);
        }
        params.rewardee.sendMessage(ChatColor.GREEN + "You got an assorted planter set.");
        return rewards;
    }, params -> {
        ItemStack item = new ItemStack(Material.SAPLING);
        item = Utility.setName(item, ChatColor.DARK_GREEN + "An Assorted Planter Set");
        return item;
    }),

    MONEY (params -> {
        params.rewardee.getServer().dispatchCommand(params.rewardee.getServer().getConsoleSender(), "eco give " + params.rewardee.getName() + " " + params.amountToGive);
        params.rewardee.sendMessage("You got $" + params.amountToGive + "!");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.GOLD_INGOT, 1);
        item = Utility.setName(item, ChatColor.DARK_AQUA + "$" + params.amountToGive);
        return item;
    }),

    INVINCIBILITY (params -> {
        params.rewardee.addPotionEffect(new PotionEffect(PotionEffectType.REGENERATION, 100, 4));
        params.rewardee.addPotionEffect(new PotionEffect(PotionEffectType.FIRE_RESISTANCE, params.amountToGive*20, 5));
        params.rewardee.addPotionEffect(new PotionEffect(PotionEffectType.DAMAGE_RESISTANCE, params.amountToGive*20, 5));
        params.rewardee.sendMessage("You got " + params.amountToGive + " Seconds of Invincibility!");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.POTION, 1);
        ItemMeta meta = item.getItemMeta();
        meta.setDisplayName(ChatColor.GOLD + "" + params.amountToGive + " Seconds of Invincibility");
        PotionMeta pmeta = (PotionMeta) meta;
        pmeta.addCustomEffect(new PotionEffect(PotionEffectType.FIRE_RESISTANCE, 1, 1), true);
        item.setItemMeta(meta);
        return item;
    }),

    MASSIVE_DAMAGE (params -> {
        params.rewardee.addPotionEffect(new PotionEffect(PotionEffectType.INCREASE_DAMAGE, params.amountToGive*1200, 6));
        if (params.amountToGive == 1)
            params.rewardee.sendMessage(ChatColor.DARK_RED + "You got 1 Minute of Massive Damage!");
        else
            params.rewardee.sendMessage(ChatColor.DARK_RED + "You got " + params.amountToGive + " Minutes of Massive Damage!");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.POTION, 1);
        ItemMeta meta = item.getItemMeta();
        if (params.amountToGive == 1)
            meta.setDisplayName(ChatColor.DARK_RED + "1 Minute of Massive Damage");
        else
            meta.setDisplayName(ChatColor.DARK_RED + "" + params.amountToGive + " Minutes of Massive Damage");
        PotionMeta pmeta = (PotionMeta) meta;
        pmeta.addCustomEffect(new PotionEffect(PotionEffectType.INCREASE_DAMAGE, 1, 1), true);
        item.setItemMeta(meta);
        return item;
    }),

    MASSIVE_HEALTH (params -> {
        params.rewardee.addPotionEffect(new PotionEffect(PotionEffectType.REGENERATION, 160, 8));
        params.rewardee.addPotionEffect(new PotionEffect(PotionEffectType.HEALTH_BOOST, params.amountToGive*72000, 19));
        if (params.amountToGive == 1)
            params.rewardee.sendMessage(ChatColor.RED + "You got 1 Hour of Massive Health!");
        else
            params.rewardee.sendMessage(ChatColor.RED + "You got " + params.amountToGive + " Hours of Massive Health!");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.POTION, 1);
        ItemMeta meta = item.getItemMeta();
        if (params.amountToGive == 1)
            params.rewardee.sendMessage(ChatColor.RED + "1 Hour of Massive Health");
        else
            meta.setDisplayName(ChatColor.RED + "" + params.amountToGive + " Hours of Massive Health");
        PotionMeta pmeta = (PotionMeta) meta;
        pmeta.addCustomEffect(new PotionEffect(PotionEffectType.REGENERATION, 1, 1), true);
        item.setItemMeta(meta);
        return item;
    }),

    INDIVIDUAL_NUGGETS (params -> {
        ArrayList<ItemStack> nuggets = new ArrayList<>();
        for (int i=0;i<params.amountToGive;i++)
            nuggets.add(new ItemStack(Material.GOLD_NUGGET, 1));
        params.rewardee.sendMessage(ChatColor.DARK_AQUA + "You got " + params.amountToGive + " individually placed gold nuggets!");
        return nuggets;
    }, params -> {
        ItemStack item = new ItemStack(Material.GOLD_NUGGET);
        if (params.amountToGive > 1)
            item = Utility.setName(item, ChatColor.DARK_AQUA + "" + params.amountToGive + " Individually Placed Gold Nuggets");
        else
            item = Utility.setName(item, ChatColor.DARK_AQUA + "1 Gold Nugget");
        return item;
    }),

    FIREWORKS_SHOW (params -> {
        for (int i=0;i<params.amountToGive;i++)
        {
            Firework firework = (Firework) params.chestBlock.getWorld().spawnEntity(params.chestBlock.getLocation().add(new Vector(Math.random(), 1, Math.random())), EntityType.FIREWORK);
            FireworkMeta meta = firework.getFireworkMeta();
            meta.setPower(Utility.randomInt(1, 4));
            FireworkEffect.Type[] effectTypes = FireworkEffect.Type.values();
            FireworkEffect effect = FireworkEffect.builder()
                    .flicker(Math.random() > 0.5)
                    .trail(Math.random() > 0.5)
                    .with(effectTypes[Utility.randomInt(0,effectTypes.length)])
                    .withColor(Color.fromRGB(Utility.randomInt(128,255), Utility.randomInt(128,255), Utility.randomInt(128,255)))
                    .build();
            meta.addEffect(effect);
            firework.setFireworkMeta(meta);
        }
        params.rewardee.sendMessage(ChatColor.GOLD + "You got a fireworks show! Yaayyy!!");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.FIREWORK);
        item = Utility.setName(item, ChatColor.GOLD + "A Fireworks Show");
        return item;
    }),

    NOTHING (params -> {
        params.rewardee.sendMessage(ChatColor.DARK_GRAY + "You got nothing.");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.THIN_GLASS);
        item = Utility.setName(item, ChatColor.DARK_GRAY + "Nothing");
        return item;
    });

    private PrizeAction action;
    private PrizeVisual visualisation;
    Prize(PrizeAction action, PrizeVisual visualisation)
    {
        this.action = action;
        this.visualisation = visualisation;
    }

    public void giveReward(LootCrate plugin, Player rewardee, int amount, Block chestBlock)
    {
        if (chestBlock.getType() != Material.CHEST)
            return;
        List<ItemStack> rewardItemsRaw = action.enactReward(new RewardActionParameter(plugin, rewardee, amount, chestBlock));
        if (rewardItemsRaw == null)
            rewardItemsRaw = new ArrayList<>();
        ArrayList<ItemStack> rewardItems = new ArrayList<>();
        for (int i=0;i<rewardItemsRaw.size();i++)
        {
            while (rewardItemsRaw.get(i).getAmount() > rewardItemsRaw.get(i).getType().getMaxStackSize())
            {
                ItemStack newStack = rewardItemsRaw.get(i).clone();
                newStack.setAmount(newStack.getType().getMaxStackSize());
                rewardItems.add(newStack);
                rewardItemsRaw.get(i).setAmount(rewardItemsRaw.get(i).getAmount() - rewardItemsRaw.get(i).getType().getMaxStackSize());
            }
            rewardItems.add(rewardItemsRaw.get(i));
        }
        Chest chest = (Chest) chestBlock.getState();
        int offsetdirection = -1;
        for (int i=0;i<rewardItems.size();i++)
        {
            int index = chest.getInventory().getSize()/2 + ((int)(i/2.0f + 0.5)*offsetdirection);
            chest.getInventory().setItem(index, rewardItems.get(i));
            offsetdirection*=-1;
        }
    }

    public ItemStack getVisualisation(LootCrate plugin, Player rewardee, int amount, Block chestBlock)
    {
        return visualisation.getVisualisation(new RewardActionParameter(plugin, rewardee, amount, chestBlock));
    }

    public String getLoreTag()
    {
        return ChatColor.BLACK + toString().toLowerCase();
    }

    public class RewardActionParameter
    {
        public LootCrate plugin;
        public Player rewardee;
        public int amountToGive;
        public Block chestBlock;

        public RewardActionParameter(LootCrate plugin, Player rewardee, int amountToGive, Block chestBlock)
        {
            this.plugin = plugin;
            this.rewardee = rewardee;
            this.amountToGive = amountToGive;
            this.chestBlock = chestBlock;
        }
    }

    interface PrizeVisual
    {
        ItemStack getVisualisation(RewardActionParameter parameters);
    }

    interface PrizeAction
    {
        List<ItemStack> enactReward(RewardActionParameter parameters);
    }
}