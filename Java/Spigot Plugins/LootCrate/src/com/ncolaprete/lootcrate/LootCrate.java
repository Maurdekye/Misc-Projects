package com.ncolaprete.lootcrate;

import net.minecraft.server.v1_9_R1.*;
import org.bukkit.*;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.block.Chest;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.command.ConsoleCommandSender;
import org.bukkit.configuration.Configuration;
import org.bukkit.configuration.ConfigurationSection;
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
import org.bukkit.event.player.PlayerItemHeldEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.FireworkMeta;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.inventory.meta.PotionMeta;
import org.bukkit.material.MaterialData;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;
import org.bukkit.util.BlockIterator;
import org.bukkit.util.Vector;

import java.lang.reflect.Field;
import java.util.*;
import java.util.stream.Collectors;

public class LootCrate extends JavaPlugin implements Listener, CommandExecutor{

    public ArrayList<CrateKey> crateKeys;
    private ArrayList<CrateLayout> crateLayouts;
    private ArrayList<Crate> cratePositions;
    public HashMap<UUID, Long> tempCreativeTimestamps;
    private ArrayList<Job> activeJobs;

    private CustomConfig crateKeyConfig;
    private CustomConfig crateLayoutConfig;
    private CustomConfig cratePositionConfig;
    public CustomConfig tempCreativeTimestampConfig;
    private CustomConfig optionsConfig;

    private ConsoleCommandSender csend = getServer().getConsoleSender();

    // config variables

    public int MaxBlocksPerFell;
    public int TreefellerSpeed;
    public boolean CanFellTrees;
    public boolean CanFellMushrooms;

    public int MaxBlocksPerTransmogrophy;
    public int TransmogiphySpeed;

    public int TerramorpherSize;

    public int GigaDrillBreakerSize;

    // Overridden Methods

    public void onEnable()
    {
        // register listeners / repeating events
        getServer().getPluginManager().registerEvents(this, this);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::updateJobs, 0, 1);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::checkAllPlayersForSpecialItems, 0, 20);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::checkForCreativeTimeUp, 0, 1200);

        // initialize arrays
        crateKeys = new ArrayList<>();
        crateLayouts = new ArrayList<>();
        cratePositions = new ArrayList<>();
        tempCreativeTimestamps = new HashMap<>();
        activeJobs = new ArrayList<>();

        // load up sample example files
        CustomConfig crateKeysSampleFile = new CustomConfig(this, "crate_keys_sample_file.yml");
        CustomConfig crateLayoutsSampleFile = new CustomConfig(this, "crate_layouts_sample_file.yml");
        crateKeysSampleFile.getConfig().options().copyDefaults(true);
        crateLayoutsSampleFile.getConfig().options().copyDefaults(true);
        crateKeysSampleFile.saveConfig();
        crateLayoutsSampleFile.saveConfig();

        // load up configs
        crateKeyConfig = new CustomConfig(this, "crate_keys.yml");
        crateLayoutConfig = new CustomConfig(this, "crate_layouts.yml");
        cratePositionConfig = new CustomConfig(this, "crate_positions.yml");
        tempCreativeTimestampConfig = new CustomConfig(this, "temp_creatives.yml");
        optionsConfig = new CustomConfig(this, "lootcrate_config.yml");

        crateKeyConfig.getConfig().options().copyDefaults(true);
        crateLayoutConfig.getConfig().options().copyDefaults(true);
        optionsConfig.getConfig().options().copyDefaults(true);

        crateKeyConfig.saveConfig();
        crateLayoutConfig.saveConfig();
        cratePositionConfig.saveConfig();
        tempCreativeTimestampConfig.saveConfig();
        optionsConfig.saveConfig();

        // load in crate keys
        for (String key : crateKeyConfig.getConfig().getKeys(false))
        {
            ConfigurationSection keysection = crateKeyConfig.getConfig().getConfigurationSection(key);
            String type = key;
            try {
                Prize.valueOf(type.toUpperCase());
                csend.sendMessage(ChatColor.RED + "Error! Key name '" + type + "' conflicts with existing prize!");
                continue;
            } catch (Exception e){};
            if (type.equalsIgnoreCase("_no_key"))
            {
                csend.sendMessage(ChatColor.RED + "Error: The key name '_no_key' is disallowed.");
                continue;
            }
            String materialname = keysection.getString("material", "tripwire_hook");
            Material material = Material.getMaterial(materialname.toUpperCase());
            if (material == null)
            {
                csend.sendMessage(ChatColor.RED + "Error! '" + materialname + "' is not a valid block or item.");
                continue;
            }
            String name = keysection.getString("name", ChatColor.RED + "Undefined Key");
            name = ChatColor.translateAlternateColorCodes('?', name);
            String lorestring = keysection.getString("description", "");
            lorestring = ChatColor.translateAlternateColorCodes('?', lorestring);
            List<String> lore = Arrays.asList(lorestring.split("\\\\n"));
            crateKeys.add(new CrateKey(type, material, name, lore));
        }

        if (crateKeys.size() == 0)
        {
            csend.sendMessage(ChatColor.RED + "Error; No crate keys detected! Add them to crate_keys.yml and reload the plugin.");
        }

        // load in crate layouts
        for (String key : crateLayoutConfig.getConfig().getKeys(false))
        {
            ConfigurationSection cratesection = crateLayoutConfig.getConfig().getConfigurationSection(key);
            String type = key;
            String printname = cratesection.getString("name");
            printname = ChatColor.translateAlternateColorCodes('?', printname);
            String reqKeyName = cratesection.getString("required_key", "_no_key");
            double spawnChance = cratesection.getDouble("spawn_chance");
            boolean broadcast = cratesection.getBoolean("broadcast_on_drop", false);
            CrateKey reqKey = null;
            if (!reqKeyName.equalsIgnoreCase("_no_key"))
            {
                for (CrateKey ck : crateKeys)
                {
                    if (ck.type.equalsIgnoreCase(reqKeyName))
                    {
                        reqKey = ck;
                        break;
                    }
                }
                if (reqKey == null)
                {
                    csend.sendMessage(ChatColor.RED + "Error! crate key '" + reqKeyName + "' not found in crate_keys.yml!");
                    continue;
                }
            }
            ArrayList<Reward> rewardList = new ArrayList<>();
            for (String rewardKey : cratesection.getConfigurationSection("rewards").getKeys(false))
            {
                ConfigurationSection rewardsection = cratesection.getConfigurationSection("rewards." + rewardKey);
                String prizeName = rewardsection.getString(".prize");
                Prize prize = null;
                double rewardChance;
                int amount;
                int keyPrizeIndex = 0;
                try {
                    prize = Prize.valueOf(prizeName.toUpperCase());
                    if (prize == Prize._CRATE_KEY)
                    {
                        csend.sendMessage(ChatColor.RED + "Error: cannot reference the '_crate_key' prize directly. Use key names as prizes instead.");
                        continue;
                    }
                } catch (Exception e) {
                    for (CrateKey ck : crateKeys)
                    {
                        if (ck.type.equalsIgnoreCase(prizeName))
                        {
                            prize = Prize._CRATE_KEY;
                            break;
                        }
                        keyPrizeIndex++;
                    }
                    if (prize == null)
                    {
                        csend.sendMessage(ChatColor.RED + "Error! Unknown prize: " + prizeName);
                        continue;
                    }
                }
                try {
                    rewardChance = Double.parseDouble(rewardsection.getString("chance"));
                } catch (Exception e) {
                    csend.sendMessage(ChatColor.RED + "Error! '" + rewardsection.getString("chance") + "' is not a number!");
                    continue;
                }
                try {
                    amount = Integer.parseInt(rewardsection.getString("amount"));
                } catch (Exception e) {
                    csend.sendMessage(ChatColor.RED + "Error! '" + rewardsection.getString("amount") + "' is not an integer!");
                    continue;
                }

                if (prize == Prize._CRATE_KEY)
                {
                    amount = crateKeys.size() * (amount - 1) + keyPrizeIndex;
                }

                rewardList.add(new Reward(prize, amount, rewardChance));
            }
            crateLayouts.add(new CrateLayout(printname, type, spawnChance, reqKey, broadcast, rewardList));
        }

        if (crateLayouts.size() == 0)
        {
            csend.sendMessage(ChatColor.RED + "Error; No crate layouts detected! Add them to crate_layouts.yml and reload the plugin.");
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
        for (String key : tempCreativeTimestampConfig.getConfig().getKeys(false))
        {
            tempCreativeTimestamps.put(UUID.fromString(key), tempCreativeTimestampConfig.getConfig().getLong(key));
        }

        // load in config options
        ConfigurationSection treefellerCfg = optionsConfig.getConfig().getConfigurationSection("treefeller");
        MaxBlocksPerFell = treefellerCfg.getInt("maxblocks", 256);
        TreefellerSpeed = treefellerCfg.getInt("speed", 1);
        CanFellTrees = treefellerCfg.getBoolean("trees", true);
        CanFellMushrooms = treefellerCfg.getBoolean("mushrooms", true);

        ConfigurationSection transmogrifierCfg = optionsConfig.getConfig().getConfigurationSection("transmogrifier");
        MaxBlocksPerTransmogrophy = transmogrifierCfg.getInt("maxblocks", 64);
        TransmogiphySpeed = transmogrifierCfg.getInt("speed", 3);

        ConfigurationSection terramorpherCfg = optionsConfig.getConfig().getConfigurationSection("terramorpher");
        TerramorpherSize = terramorpherCfg.getInt("size");

        ConfigurationSection gigadrillbreakerCfg = optionsConfig.getConfig().getConfigurationSection("giga_drill_breaker");
        GigaDrillBreakerSize = gigadrillbreakerCfg.getInt("size");

        // startup random crate dropper
        ConfigurationSection cratespawningSection = optionsConfig.getConfig().getConfigurationSection("cratespawning");
        if (cratespawningSection.getBoolean("spawncrates")) {
            int interval = cratespawningSection.getInt("interval", 300) * 20;
            final int radius = cratespawningSection.getInt("radius", 1000);
            final boolean broadcast = cratespawningSection.getBoolean("broadcast", false);
            getServer().getScheduler().scheduleSyncRepeatingTask(this, () -> dropRandomCrate(Utility.getDefaultSpawn(this), radius), 0, interval);
        }
    }

    public void onDisable()
    {
        cratePositionConfig.saveConfig();
        tempCreativeTimestampConfig.saveConfig();
    }

    public boolean onCommand(CommandSender sender, Command command, String label, String[] args)
    {
        Player ply = sender instanceof Player ? (Player) sender : null;

        // givekey
        if (command.getName().equalsIgnoreCase("givekey"))
        {

            // Check if there are any crate keys loaded
            if (crateKeys.size() == 0)
            {
                sender.sendMessage(ChatColor.RED + "Error: No crate keys loaded. Add them to crate_keys.yml and reload the plugin.");
                return true;
            }

            // Check if enough arguments were provided
            if (args.length == 0)
            {
                StringBuilder rewardslist = new StringBuilder();
                rewardslist.append("Available crate keys are: ");
                for (CrateKey k : crateKeys)
                    rewardslist.append(k.type.toLowerCase() + ", ");
                ply.sendMessage(rewardslist.substring(0, rewardslist.length()-2));
                return false;
            }

            // Find crate key to give
            CrateKey key = null;
            for (CrateKey k : crateKeys)
            {
                if (k.type.equalsIgnoreCase(args[0]))
                {
                    key = k;
                    break;
                }
            }

            if (key == null)
            {
                sender.sendMessage(ChatColor.RED + "Could not find crate key '" + args[0] + "'.");
                return true;
            }

            // Find amount to give
            int amount = 1;
            if (args.length >= 2)
            {
                try {
                    amount = Integer.parseInt(args[1]);
                } catch (Exception e) {
                    sender.sendMessage(ChatColor.RED + "'" + args[1] + "' is not a number.");
                    return true;
                }
            }

            // Find player to give key to
            Player target = ply;
            if (args.length >= 3)
            {
                String targetname = "";
                for (int i=2;i<args.length;i++)
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

            // Check if sender is player and target is unset
            if (target == null)
            {
                sender.sendMessage(ChatColor.RED + "You must be a player to use this command on yourself");
                return true;
            }
            else
                target = ply;

            // Give the key(s)
            ItemStack keyStack = key.getKey(false);
            keyStack.setAmount(amount);
            for (ItemStack item : Utility.separateItemStacks(Collections.singletonList(keyStack)))
                target.getInventory().addItem(item);
        }

        // givecrate
        else if (command.getName().equalsIgnoreCase("givecrate"))
        {

            // Check if there are any crate layouts loaded
            if (crateLayouts.size() == 0)
            {
                sender.sendMessage(ChatColor.RED + "Error: No crate layouts loaded. Add them to crate_layouts.yml and reload the plugin.");
                return true;
            }

            // Check if enough arguments were provided
            if (args.length == 0)
            {
                StringBuilder rewardslist = new StringBuilder();
                rewardslist.append("Available crate layouts are: ");
                for (CrateLayout l : crateLayouts)
                    rewardslist.append(l.type.toLowerCase() + ", ");
                ply.sendMessage(rewardslist.substring(0, rewardslist.length()-2));
                return false;
            }

            // Find crate layout to use
            String type = args[0];
            CrateLayout layout = null;
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

            // Find amount to give
            int amount = 1;
            if (args.length >= 2)
            {
                try {
                    amount = Integer.parseInt(args[1]);
                } catch (Exception e) {
                    sender.sendMessage(ChatColor.RED + "'" + args[1] + "' is not a number.");
                    return true;
                }
            }

            // Find player to give crate to
            Player target = ply;
            if (args.length >= 3)
            {
                String targetname = "";
                for (int i=2;i<args.length;i++)
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

            // Check if sender is player and target is unset
            if (target == null)
            {
                sender.sendMessage(ChatColor.RED + "You must be a player to use this command on yourself");
                return true;
            }
            else
                target = ply;

            // Give the crate(s)
            ItemStack crateStack = layout.getItemstack();
            crateStack.setAmount(amount);
            for (ItemStack item : Utility.separateItemStacks(Collections.singletonList(crateStack)))
                target.getInventory().addItem(item);
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

            // Check if enough arguments were provided
            if (args.length == 0)
            {
                StringBuilder rewardslist = new StringBuilder();
                rewardslist.append("Available rewards are: ");
                for (Prize p : Prize.values())
                    rewardslist.append(p.name().toLowerCase() + ", ");
                ply.sendMessage(rewardslist.substring(0, rewardslist.length()-2));
                return false;
            }

            // Find prize type to give
            Prize type;
            try {
                type = Prize.valueOf(args[0].toUpperCase());
            } catch (Exception e) {
                sender.sendMessage(ChatColor.RED + "Could not find prize " + args[0] + ".");
                return true;
            }

            // Find amount to give
            int amount = 1;
            if (args.length >= 2)
            {
                try {
                    amount = Integer.parseInt(args[1]);
                } catch (Exception e) {
                    sender.sendMessage(ChatColor.RED + "'" + args[1] + "' is not a number.");
                    return true;
                }
            }

            // Find chest to give from
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

            // Give the prize
            type.giveReward(this, ply, amount, chestBlock);
        }
        return true;
    }

    // Repeating Runnables

    private void updateJobs()
    {
        for (Job job : activeJobs)
            job.update();
        activeJobs.removeIf(j -> j.isDone());
    }

    private void checkAllPlayersForSpecialItems()
    {
        for (Player ply : getServer().getOnlinePlayers())
        {
            checkPlayerForSpecialItem(ply);
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
                tempCreativeTimestampConfig.getConfig().set(plyId.toString(), null);
                tempCreativeTimestampConfig.saveConfig();
            }
            else
            {
                ply.setGameMode(GameMode.CREATIVE);
            }
        }
    }

    // Event Handlers

    @EventHandler
    public void playerInteract(PlayerInteractEvent ev)
    {
        // manage opening of crates
        if (ev.getAction() == Action.RIGHT_CLICK_BLOCK && !ev.getPlayer().isSneaking())
        {
            Block block = ev.getClickedBlock();
            if (block.getType() != Material.CHEST)
                return;
            Crate crateToOpen = null;
            for (Crate crate : cratePositions)
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
                if (!crateToOpen.layout.isFree())
                {
                    if (handItem.getAmount() > 1)
                        handItem.setAmount(handItem.getAmount() - 1);
                    else
                        ev.getPlayer().getInventory().remove(handItem);
                    crateToOpen.unlockAndGivePrize(this, ev.getPlayer());
                    removeCrate(crateToOpen);
                }
            }
            else
            {
                ev.setCancelled(true);
                ev.getPlayer().openInventory(crateToOpen.showContents(this, ev.getPlayer()));
            }
        }

        // Transmogrifier
        if (ev.getAction() == Action.LEFT_CLICK_BLOCK &&
                Utility.itemHasLoreLine(ev.getPlayer().getInventory().getItemInMainHand(), Prize.TRANSMOGRIFIER.getLoreTag()))
        {
            ItemStack offhandItem = ev.getPlayer().getInventory().getItemInOffHand();
            if (offhandItem == null)
                return;
            if (!offhandItem.getType().isBlock())
                return;
            if (!offhandItem.getType().isSolid())
                return;
            if (!ev.getClickedBlock().getType().isSolid())
                return;
            if (ev.getClickedBlock().getType() == offhandItem.getType() && ev.getClickedBlock().getData() == offhandItem.getDurability())
                return;
            activeJobs.add(new TransmogrificationJob(ev.getClickedBlock(), ev.getPlayer(), TransmogiphySpeed, MaxBlocksPerTransmogrophy));
        }
    }

    @EventHandler
    public void blockBreak(BlockBreakEvent ev)
    {
        // Treefeller Chainsaw
        if (Utility.itemHasLoreLine(ev.getPlayer().getInventory().getItemInMainHand(), Prize.TREEFELLER_CHAINSAW.getLoreTag()))
        {
            if (new TreefellerJob().getValidBlocks().contains(ev.getBlock().getType()) && CanFellTrees)
            {
                activeJobs.add(new TreefellerJob(ev.getBlock(), TreefellerSpeed, MaxBlocksPerFell));
            }
            else if (new ShroomFellerJob().getValidBlocks().contains(ev.getBlock().getType()) && CanFellMushrooms)
            {
                activeJobs.add(new ShroomFellerJob(ev.getBlock(), TreefellerSpeed, MaxBlocksPerFell));
            }
        }

        // Terramorpher
        if (Utility.itemHasLoreLine(ev.getPlayer().getInventory().getItemInMainHand(), Prize.TERRAMORPHER.getLoreTag()))
        {
            if (!Utility.isCorrectTool(Material.DIAMOND_SPADE, ev.getBlock().getType()))
                return;
            activeJobs.add(new TerramorpherJob(ev.getBlock(), Utility.getBlockFaceIsLookingAt(ev.getPlayer()), TerramorpherSize));
        }

        // Giga Drill Breaker
        if (Utility.itemHasLoreLine(ev.getPlayer().getInventory().getItemInMainHand(), Prize.GIGA_DRILL_BREAKER.getLoreTag()))
        {
            if (!Utility.isCorrectTool(Material.DIAMOND_PICKAXE, ev.getBlock().getType()))
                return;
            activeJobs.add(new TerramorpherJob(ev.getBlock(), Utility.getBlockFaceIsLookingAt(ev.getPlayer()), GigaDrillBreakerSize));
        }

        // Manage picking up of crates
        if (ev.getBlock().getType() == Material.CHEST)
        {
            for (int i = 0; i < cratePositions.size(); i++) {
                if (cratePositions.get(i).location.equals(ev.getBlock()))
                {
                    ev.setCancelled(true);
                    ItemStack crateDrop = cratePositions.get(i).layout.getItemstack();
                    ev.getBlock().setType(Material.AIR);
                    ev.getBlock().getWorld().dropItemNaturally(ev.getBlock().getLocation().add(0.5, 0.5, 0.5), crateDrop);
                    removeCrate(i);
                    break;
                }
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
            if (Utility.itemHasLoreLine(ev.getItemInHand(), l.getLoreTag()))
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
                for (Crate c : cratePositions)
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
    public void playerItemHeld(PlayerItemHeldEvent ev)
    {
        getServer().getScheduler().scheduleSyncDelayedTask(this, () -> checkPlayerForSpecialItem(ev.getPlayer()), 1);
    }

    @EventHandler
    public void projectileLaunch(ProjectileLaunchEvent ev)
    {
        if (ev.getEntity() instanceof Arrow && ev.getEntity().getShooter() instanceof Player)
        {
            Player ply = (Player) ev.getEntity().getShooter();
            if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), Prize.VEILSTRIKE_BOW.getLoreTag()))
            {
                ev.getEntity().setVelocity(ev.getEntity().getVelocity().multiply(4));
            }
        }
    }

    // Other Methods

    private void removeCrate(Crate c)
    {
        cratePositionConfig.getConfig().set(Utility.serializeLocation(c.location.getLocation()), null);
        cratePositionConfig.saveConfig();
        cratePositions.remove(c);
    }

    private void removeCrate(int index)
    {
        cratePositionConfig.getConfig().set(Utility.serializeLocation(cratePositions.get(index).location.getLocation()), null);
        cratePositionConfig.saveConfig();
        cratePositions.remove(index);
    }

    private void addCrate(Crate c)
    {
        cratePositions.add(c);
        cratePositionConfig.getConfig().set(Utility.serializeLocation(c.location.getLocation()), c.layout.type);
        cratePositionConfig.saveConfig();
    }

    private void dropRandomCrate(Location center, double radius)
    {
        Location droplocation;
        Block newChest;
        do {
            droplocation = center.add(Utility.randomInsideUnitCircle().multiply(radius));
            newChest = Utility.getHighestSolidBlock(center.getWorld(), droplocation.getBlockX(), droplocation.getBlockZ());
        } while (newChest.getLocation().getY() >= droplocation.getWorld().getMaxHeight());
        ArrayList<Double> weights = new ArrayList<>();
        for (CrateLayout l : crateLayouts)
            weights.add(l.spawnChance);
        CrateLayout layout = crateLayouts.get(Utility.randomWeightedIndex(weights));
        addCrate(new Crate(newChest.getRelative(BlockFace.UP), layout));
        if (layout.shouldBroadcast)
        {
            getServer().broadcastMessage("A " + layout.printname + ChatColor.RESET + " has dropped at " + ChatColor.GOLD + newChest.getX() + ", " + newChest.getZ() + ChatColor.RESET + "!");
        }
        csend.sendMessage(layout.printname + ChatColor.RESET + " spawned at " + Utility.formatVector(newChest.getLocation().toVector()));
    }

    private void checkPlayerForSpecialItem(Player ply)
    {
        // Frostspark Cleats
        if (Utility.itemHasLoreLine(ply.getInventory().getBoots(), Prize.FROSTSPARK_CLEATS.getLoreTag()))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.SPEED, 40, 2), true);
            ply.addPotionEffect(new PotionEffect(PotionEffectType.JUMP, 40, 1), true);
        }

        // Lucky Trousers
        if (Utility.itemHasLoreLine(ply.getInventory().getLeggings(), Prize.LUCKY_TROUSERS.getLoreTag()))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.LUCK, 40, 2), true);
        }

        // Knackerbreaker Chesterplate
        if (Utility.itemHasLoreLine(ply.getInventory().getChestplate(), Prize.KNACKERBREAKER_CHESTERPLATE.getLoreTag()))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.ABSORPTION, 40, 0), true);
        }

        // Hydrodyne Helmet
        if (Utility.itemHasLoreLine(ply.getInventory().getHelmet(), Prize.HYDRODYNE_HELMET.getLoreTag()))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.NIGHT_VISION, 250, 0), true);
            ply.addPotionEffect(new PotionEffect(PotionEffectType.WATER_BREATHING, 40, 0), true);
        }

        // Giga Drill Breaker
        if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), Prize.GIGA_DRILL_BREAKER.getLoreTag()))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.FAST_DIGGING, 25, 3), true);
        }

        // Unyielding Battersea
        if ((Utility.itemHasLoreLine(ply.getInventory().getItemInOffHand(), Prize.UNYIELDING_BATTERSEA.getLoreTag()) ||
                Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), Prize.UNYIELDING_BATTERSEA.getLoreTag())) &&
                ply.isBlocking())
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.DAMAGE_RESISTANCE, 40, 0), true);
            ply.addPotionEffect(new PotionEffect(PotionEffectType.FIRE_RESISTANCE, 40, 0), true);
        }

        // Veilstrike Bow
        if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), Prize.VEILSTRIKE_BOW.getLoreTag()))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.INVISIBILITY, 25, 0), true);
        }

        // Heaven's Blade
        if (Utility.itemHasLoreLine(ply.getInventory().getItemInMainHand(), Prize.HEAVENS_BLADE.getLoreTag()))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.INCREASE_DAMAGE, 25, 4), true);
        }
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

    static Vector randomInsideUnitCircle()
    {
        double x, y;
        do {
            x = Math.random() * 2 - 1;
            y = Math.random() * 2 - 1;
        } while (x*x + y*y > 1);
        return new Vector(x, 0, y);
    }

    static int randomWeightedIndex(List<Double> weights)
    {
        double sum = 0;
        for (double f : weights)
            sum += f;
        double rand = Math.random() * sum;
        for (int i=0;i<weights.size();i++)
        {
            if (rand < weights.get(i))
                return i;
            rand -= weights.get(i);
        }
        return weights.size() - 1;
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
        String[] parts = serial.split("\\?");
        World world = server.getWorld(parts[0]);
        int x = Integer.parseInt(parts[1]);
        int y = Integer.parseInt(parts[2]);
        int z = Integer.parseInt(parts[3]);
        return world.getBlockAt(x, y, z).getLocation();
    }

    static Location getDefaultSpawn(JavaPlugin plugin)
    {
        return plugin.getServer().getWorlds().get(0).getSpawnLocation();
    }

    static Block getHighestSolidBlock(World world, int x, int z)
    {
        Location start = world.getHighestBlockAt(x, z).getLocation();
        BlockIterator iter = new BlockIterator(world, start.toVector().add(new Vector(0.5, 0.5, 0.5)), new Vector(0, -1, 0), 0, 255);
        Block highestSolid;
        do {
            highestSolid = iter.next();
        } while (!highestSolid.getType().isSolid() && !highestSolid.isLiquid() && iter.hasNext());
        if (highestSolid.getY() >= world.getMaxHeight())
            return null;
        if (!iter.hasNext())
            return null;
        return highestSolid;
    }

    static String formatVector(Vector v)
    {
        StringBuilder b = new StringBuilder();
        b.append("(");
        b.append(v.getBlockX());
        b.append(", ");
        b.append(v.getBlockY());
        b.append(", ");
        b.append(v.getBlockZ());
        b.append(")");
        return b.toString();
    }

    static List<ItemStack> separateItemStacks(List<ItemStack> items)
    {
        ArrayList<ItemStack> separatedItems = new ArrayList<>();
        for (int i=0;i<items.size();i++)
        {
            while (items.get(i).getAmount() > items.get(i).getType().getMaxStackSize())
            {
                ItemStack newStack = items.get(i).clone();
                newStack.setAmount(newStack.getType().getMaxStackSize());
                separatedItems.add(newStack);
                items.get(i).setAmount(items.get(i).getAmount() - items.get(i).getType().getMaxStackSize());
            }
            separatedItems.add(items.get(i));
        }
        return separatedItems;
    }

    static List<Block> getSurroundingBlocks(Block block, boolean sides, boolean diagonals, boolean corners)
    {
        ArrayList<Block> surrounds = new ArrayList<>();
        List<BlockFace> cardinalFaces = Arrays.asList(BlockFace.NORTH, BlockFace.SOUTH, BlockFace.EAST, BlockFace.WEST, BlockFace.UP, BlockFace.DOWN);
        if (sides)
        {
            surrounds.addAll(cardinalFaces.stream().map(block::getRelative).collect(Collectors.toList()));
            surrounds.add(block.getRelative(BlockFace.UP));
            surrounds.add(block.getRelative(BlockFace.DOWN));
        }
        if (diagonals)
        {
            List<BlockFace> diagonalFaces = Arrays.asList(BlockFace.NORTH_EAST, BlockFace.NORTH_WEST, BlockFace.SOUTH_EAST, BlockFace.SOUTH_WEST);
            for (BlockFace face : cardinalFaces)
            {
                surrounds.add(block.getRelative(face).getRelative(BlockFace.UP));
                surrounds.add(block.getRelative(face).getRelative(BlockFace.DOWN));
            }
            surrounds.addAll(diagonalFaces.stream().map(block::getRelative).collect(Collectors.toList()));
        }
        if (corners)
        {
            for (BlockFace fa : Arrays.asList(BlockFace.NORTH, BlockFace.SOUTH))
            {
                for (BlockFace fb : Arrays.asList(BlockFace.EAST, BlockFace.WEST))
                {
                    surrounds.addAll(Arrays.asList(BlockFace.UP, BlockFace.DOWN).stream().map(fc -> block.getRelative(fa).getRelative(fb).getRelative(fc)).collect(Collectors.toList()));
                }
            }
        }
        return surrounds;
    }

    static boolean isCorrectTool(Material tool, Material block)
    {
        List<Material> Pickaxes = Arrays.asList(Material.WOOD_PICKAXE, Material.STONE_PICKAXE, Material.IRON_PICKAXE, Material.DIAMOND_PICKAXE);
        List<Material> Axes = Arrays.asList(Material.WOOD_AXE, Material.STONE_AXE, Material.IRON_AXE, Material.DIAMOND_AXE);
        List<Material> Shovels = Arrays.asList(Material.WOOD_SPADE, Material.STONE_SPADE, Material.IRON_SPADE, Material.DIAMOND_SPADE);
        List<Material> validAxeBlocks = Arrays.asList(
                Material.WOOD_DOOR, Material.ACACIA_DOOR, Material.BIRCH_DOOR,
                Material.DARK_OAK_DOOR, Material.JUNGLE_DOOR, Material.SPRUCE_DOOR,
                Material.TRAP_DOOR, Material.CHEST, Material.WORKBENCH,
                Material.FENCE, Material.FENCE_GATE, Material.JUKEBOX,
                Material.WOOD, Material.LOG, Material.LOG_2, Material.BOOKSHELF,
                Material.JACK_O_LANTERN, Material.PUMPKIN, Material.SIGN_POST,
                Material.WALL_SIGN, Material.NOTE_BLOCK, Material.WOOD_PLATE,
                Material.DAYLIGHT_DETECTOR, Material.DAYLIGHT_DETECTOR_INVERTED,
                Material.HUGE_MUSHROOM_1, Material.HUGE_MUSHROOM_2, Material.VINE);
        List<Material> validShovelBlocks = Arrays.asList(
                Material.CLAY, Material.SOIL, Material.GRASS, Material.GRASS_PATH,
                Material.GRAVEL, Material.MYCEL, Material.DIRT, Material.SAND,
                Material.SOUL_SAND, Material.SNOW_BLOCK);
        if (Pickaxes.contains(tool))
        {
            return !validAxeBlocks.contains(block) && !validShovelBlocks.contains(block);
        }
        else if (Axes.contains(tool))
        {
            return validAxeBlocks.contains(block);
        }
        else if (Shovels.contains(tool))
        {
            return validShovelBlocks.contains(block);
        }
        return false;
    }

    static boolean blockInBoundsInclusive(Block minBounds, Block maxBounds, Block check)
    {
        if (!(check.getX() >= minBounds.getX() && check.getX() <= maxBounds.getX()))
            return false;
        if (!(check.getY() >= minBounds.getY() && check.getY() <= maxBounds.getY()))
            return false;
        if (!(check.getZ() >= minBounds.getZ() && check.getZ() <= maxBounds.getZ()))
            return false;
        return true;
    }

    static BlockFace getBlockFaceIsLookingAt(Player ply)
    {
        BlockIterator iter = new BlockIterator(ply);
        Block lastBlock = null;
        do {
            Block nextBlock = iter.next();
            if (lastBlock != null && nextBlock.getType() != Material.AIR)
                return lastBlock.getFace(nextBlock);
            lastBlock = nextBlock;
        } while (iter.hasNext());
        return BlockFace.SELF;
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
    double spawnChance;
    CrateKey keyRequired;
    public boolean shouldBroadcast;
    ArrayList<Reward> contents;

    public CrateLayout(String printname, String type, double spawnChance, CrateKey keyRequired, boolean shouldBroadcast, ArrayList<Reward> contents)
    {
        this.printname = printname;
        this.type = type;
        this.spawnChance = spawnChance;
        this.keyRequired = keyRequired;
        this.shouldBroadcast = shouldBroadcast;
        this.contents = contents;
    }

    public void givePrize(LootCrate plugin, Player rewardee, Block location)
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

    public Inventory showContents(LootCrate plugin, Player ply, Block chestblock)
    {
        Inventory display = Bukkit.createInventory(null, (int)Math.floor((contents.size()-1)/9.0 + 1)*9 /*lock the size to multiples of 9*/, getPrintname(true));
        double totalProbability = 0;
        for (Reward r : contents)
            totalProbability += r.rewardChance;
        for (Reward r : contents)
        {
            double chance = (r.rewardChance / totalProbability) * 10000;
            chance = Math.round(chance) / 100;
            ItemStack displayItem =  r.item.getVisualisation(plugin, ply, r.amount, chestblock);
            displayItem = Utility.addLoreLine(displayItem, ChatColor.WHITE + "%" + chance + " chance");
            display.addItem(displayItem);
        }
        return display;
    }

    public boolean isKeyValid(ItemStack key)
    {
        if (keyRequired == null)
            return true;
        return Utility.itemHasLoreLine(key, keyRequired.getLoreTag());
    }

    public String getPrintname(boolean isLocked)
    {
        if (isLocked)
            return printname + LockedTag;
        return printname + UnlockedTag;
    }

    public String getLoreTag()
    {
        return ChatColor.BLACK + type;
    }

    public ItemStack getItemstack()
    {
        ItemStack crateDrop = Utility.setName(new ItemStack(Material.CHEST), getPrintname(true));
        if (keyRequired != null)
            crateDrop = Utility.addLoreLine(crateDrop, ChatColor.RESET + "" + ChatColor.GRAY + "Requires a " + keyRequired.displayname + ChatColor.RESET + ChatColor.GRAY + " to unlock");
        crateDrop = Utility.addLoreLine(crateDrop, getLoreTag());
        return crateDrop;
    }

    public boolean isFree()
    {
        return keyRequired == null;
    }
}

class CrateKey
{
    public String type;
    public Material material;
    public String displayname;
    public List<String> lore;

    public CrateKey(String type, Material material, String displayname, List<String> lore)
    {
        this.type = type;
        this.material = material;
        this.displayname = displayname;
        this.lore = lore;
    }

    public ItemStack getKey(boolean isDisplayKey)
    {
        ItemStack key = new ItemStack(material);
        key = Utility.setName(key, displayname);
        if (isDisplayKey)
            return key;
        for (String line : lore)
            key = Utility.addLoreLine(key, line);
        key = Utility.addLoreLine(key, getLoreTag());
        return key;
    }

    public String getLoreTag()
    {
        return ChatColor.BLACK + type;
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
    _CRATE_KEY (params -> {
        if (params.plugin.crateKeys.size() == 0)
            return null;
        int amount = params.amountToGive / params.plugin.crateKeys.size() + 1;
        int keyindex = params.amountToGive % params.plugin.crateKeys.size();
        CrateKey key = params.plugin.crateKeys.get(keyindex);
        ItemStack keyItem = key.getKey(false);
        keyItem.setAmount(amount);
        if (amount == 1)
            params.rewardee.sendMessage(ChatColor.GRAY + "You got a " + key.displayname + ChatColor.RESET + ChatColor.GRAY + "!");
        else
            params.rewardee.sendMessage(ChatColor.GRAY + "You got " + amount + " " + key.displayname + "s" + ChatColor.RESET + ChatColor.GRAY + "!");
        return Collections.singletonList(keyItem);
    }, params -> {
        if (params.plugin.crateKeys.size() == 0)
            return new ItemStack(Material.TRIPWIRE_HOOK);
        int amount = params.amountToGive / params.plugin.crateKeys.size() + 1;
        int keyindex = params.amountToGive % params.plugin.crateKeys.size();
        CrateKey key = params.plugin.crateKeys.get(keyindex);
        ItemStack keyItem = key.getKey(true);
        if (amount == 1)
            return keyItem;
        return Utility.setName(keyItem, amount + " " + key.displayname + "s");
    }),

    ULTIMATE_REWARD (params -> {
        params.rewardee.setGameMode(GameMode.CREATIVE);
        params.plugin.tempCreativeTimestampConfig.getConfig().set(params.rewardee.getUniqueId().toString(), System.currentTimeMillis() + params.amountToGive * 3600000);
        params.plugin.tempCreativeTimestamps.put(params.rewardee.getUniqueId(), System.currentTimeMillis() + params.amountToGive * 3600000);
        params.plugin.tempCreativeTimestampConfig.saveConfig();
        if (params.amountToGive == 1)
            params.rewardee.sendMessage("You have received the ultimate reward: " + ChatColor.BOLD + "you may be in creative for 1 hour.");
        else
            params.rewardee.sendMessage("You have received the ultimate reward: " + ChatColor.BOLD + "you may be in creative for " + params.amountToGive + " hours.");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.COMMAND);
        return Utility.setName(item, ChatColor.UNDERLINE + "" + ChatColor.BOLD + "" + "The Ultimate Reward");
    }),

    // Armor

    FROSTSPARK_CLEATS (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_BOOTS);
        item = Utility.setName(item, ChatColor.YELLOW + "Frostspark Cleats");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The cleats grant improved mobility");
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

    // Tools

    TERRAMORPHER (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_SPADE);
        item = Utility.setName(item, ChatColor.GREEN + "" + ChatColor.BOLD + "Terramorpher");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The terramorpher digs multiple blocks at once");
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.DIG_SPEED, 3);
        params.rewardee.sendMessage(ChatColor.GREEN + "You got the " + ChatColor.BOLD + "Terramorpher!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_SPADE);
        return Utility.setName(item, ChatColor.GREEN + "" + ChatColor.BOLD + "Terramorpher");
    }),

    TRANSMOGRIFIER (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_HOE);
        item = Utility.setName(item, ChatColor.GOLD + "" + ChatColor.BOLD + "Transmogrifier");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The transmogrifier will swap blocks with those in your offhand with a left click");
        item = Utility.addLoreLine(item, ChatColor.RESET + "...you can also till soil with it.");
        item.addEnchantment(Enchantment.DURABILITY, 2);
        params.rewardee.sendMessage(ChatColor.GOLD + "You got the " + ChatColor.BOLD + "Transmogrifier!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_HOE);
        return Utility.setName(item, ChatColor.GOLD + "" + ChatColor.BOLD + "Transmogrifier");
    }),

    TREEFELLER_CHAINSAW (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_AXE);
        item = Utility.setName(item, ChatColor.DARK_GREEN + "" + ChatColor.ITALIC + "Treefeller Chainsaw");
        item = Utility.addLoreLine(item, "Let gravity do the work for you!");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The chainsaw fells entire trees with a single blow");
        item.addEnchantment(Enchantment.DURABILITY, 2);
        params.rewardee.sendMessage(ChatColor.DARK_GREEN + "You got the Treefeller Chainsaw!");
        return Collections.singletonList(item);
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_AXE);
        return Utility.setName(item, ChatColor.DARK_GREEN + "" + ChatColor.ITALIC + "Treefeller Chainsaw");
    }),

    GIGA_DRILL_BREAKER (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_PICKAXE);
        item = Utility.setName(item, ChatColor.AQUA + "" + ChatColor.BOLD + "Giga Drill Breaker");
        item = Utility.addLoreLine(item, ChatColor.AQUA + "Bust through the heavens with your Drill!");
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
        item.addEnchantment(Enchantment.ARROW_DAMAGE, 5);
        item.addEnchantment(Enchantment.ARROW_KNOCKBACK, 2);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "" + ChatColor.ITALIC + "You got the Veilstrike Bow!");
        return Arrays.asList(item, new ItemStack(Material.ARROW, params.amountToGive));
    }, params -> {
        ItemStack item = new ItemStack(Material.BOW);
        return Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.ITALIC + "Veilstrike Bow");
    }),

    HEAVENS_BLADE (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_SWORD);
        item = Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.BOLD + "Heaven's Blade");
        item = Utility.addLoreLine(item, "A godly blade that weilds incredible desctructive power");
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

    // Other rewards

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
        return Utility.setName(item, ChatColor.AQUA + "" + params.amountToGive + "xAssorted Raw Ore Blocks");
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
        return Utility.setName(item, ChatColor.AQUA + "" + params.amountToGive + "xAssorted Ores");
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
        item = Utility.setName(item, ChatColor.GREEN + "" + params.amountToGive + "xAssorted Items");
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
        item = Utility.setName(item, ChatColor.DARK_GREEN + "" + params.amountToGive + "xAssorted Plants");
        return item;
    }),

    MONEY (params -> {
        params.rewardee.getServer().dispatchCommand(params.rewardee.getServer().getConsoleSender(), "eco give " + params.rewardee.getName() + " " + params.amountToGive);
        params.rewardee.sendMessage("You got $" + params.amountToGive + "!");
        return Collections.singletonList(Utility.setName(new ItemStack(Material.PAPER), ChatColor.DARK_AQUA + "$" + params.amountToGive + " invoice"));
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
        List<ItemStack> rewardItems = Utility.separateItemStacks(rewardItemsRaw);
        Chest chest = (Chest) chestBlock.getState();
        int offsetdirection = -1;
        for (int i=0;i<rewardItems.size();i++)
        {
            int index = chest.getInventory().getSize()/2 + ((int)(i/2.0f + 0.5)*offsetdirection);
            if (rewardItems.get(i).getAmount() == 1 && rewardItems.get(i).getItemMeta() != null)
                rewardItems.set(i, Utility.addLoreLine(rewardItems.get(i), getLoreTag()));
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

interface Job
{
    void update();
    boolean isDone();
}

abstract class TreefellerJobBase implements Job
{
    Block initialBlock;
    ArrayList<Block> currentSet;
    int progressThroughSet;
    int blocksBroken;
    int breaksPerAction;
    int maxDestruction;

    public TreefellerJobBase() {}

    public TreefellerJobBase(Block initialBlock, int breaksPerAction, int maxDestruction)
    {
        this.initialBlock = initialBlock;
        this.breaksPerAction = breaksPerAction;
        this.maxDestruction = maxDestruction;
        this.currentSet = new ArrayList<>();
        this.blocksBroken = 0;
        this.progressThroughSet = 0;
        currentSet.add(initialBlock);
    }

    public void update()
    {
        for (int i=0;i<breaksPerAction;i++) {
            if (blocksBroken >= maxDestruction)
                return;
            if (progressThroughSet >= currentSet.size()) {
                progressThroughSet = 0;
                getNewSet();
            } else {
                Block cblock = currentSet.get(progressThroughSet);
                cblock.getWorld().playEffect(cblock.getLocation(), Effect.TILE_BREAK, new MaterialData(cblock.getType()));
                cblock.breakNaturally(new ItemStack(Material.DIAMOND_AXE));
                blocksBroken++;
                progressThroughSet++;
            }
        }
    }

    public void getNewSet()
    {
        ArrayList<Block> nextUpdate = new ArrayList<>();
        for (Block currentblock : currentSet)
        {
            for (Block nblock : Utility.getSurroundingBlocks(currentblock, true, true, true))
            {
                if (!getValidBlocks().contains(nblock.getType()))
                    continue;
                if (nblock.getY() < initialBlock.getY())
                    continue;
                if (nextUpdate.contains(nblock))
                    continue;
                nextUpdate.add(nblock);
            }
        }
        currentSet = nextUpdate;
    }

    public boolean isDone()
    {
        return currentSet.size() == 0 || blocksBroken >= maxDestruction;
    }

    public abstract List<Material> getValidBlocks();
}

class TreefellerJob extends TreefellerJobBase
{
    public TreefellerJob() {}

    public TreefellerJob(Block initialBlock, int breaksPerAction, int maxDestruction)
    {
        super(initialBlock, breaksPerAction, maxDestruction);
    }

    public List<Material> getValidBlocks()
    {
        return Arrays.asList(Material.LOG, Material.LOG_2, Material.LEAVES, Material.LEAVES_2);
    }
}

class ShroomFellerJob extends TreefellerJob
{
    public ShroomFellerJob() {}

    public ShroomFellerJob(Block initialBlock, int breaksPerAction, int maxDestruction)
    {
        super(initialBlock, breaksPerAction, maxDestruction);
    }

    public List<Material> getValidBlocks()
    {
        return Arrays.asList(Material.HUGE_MUSHROOM_1, Material.HUGE_MUSHROOM_2);
    }
}

class TransmogrificationJob implements Job
{
    Block initialBlock;
    Material initialMaterial;
    byte initialData;
    Player ply;
    ArrayList<Block> currentSet;
    int progressThroughSet;
    int blocksModified;
    int modificationsPerAction;
    int maxModification;
    boolean outOfBlocks = false;

    public TransmogrificationJob(Block initialBlock, Player ply, int modificationsPerAction, int maxModification)
    {
        this.initialBlock = initialBlock;
        this.initialMaterial = initialBlock.getType();
        this.initialData = initialBlock.getData();
        this.ply = ply;
        this.modificationsPerAction = modificationsPerAction;
        this.maxModification = maxModification;
        this.currentSet = new ArrayList<>();
        this.blocksModified = 0;
        this.progressThroughSet = 0;
        currentSet.add(initialBlock);
    }

    @SuppressWarnings("deprecation")
    public void update()
    {
        for (int i=0;i<modificationsPerAction;i++) {
            if (blocksModified >= maxModification)
                return;
            if (outOfBlocks)
                return;
            ItemStack offhanditem = ply.getInventory().getItemInOffHand();
            if (offhanditem == null)
                return;
            if (progressThroughSet >= currentSet.size()) {
                progressThroughSet = 0;
                getNewSet();
            } else {
                if (ply.getGameMode() != GameMode.CREATIVE)
                {
                    if (offhanditem.getAmount() == 1)
                    {
                        ply.getInventory().setItem(40, null);
                        outOfBlocks = true;
                    }
                    else
                        offhanditem.setAmount(offhanditem.getAmount() - 1);
                }
                Block cblock = currentSet.get(progressThroughSet);
                cblock.getWorld().playEffect(cblock.getLocation(), Effect.TILE_BREAK, new MaterialData(cblock.getType()));
                cblock.breakNaturally();
                cblock.setType(offhanditem.getType());
                cblock.setData((byte)offhanditem.getDurability());
                blocksModified++;
                progressThroughSet++;
            }
        }
    }

    public void getNewSet()
    {
        ArrayList<Block> nextUpdate = new ArrayList<>();
        for (Block currentblock : currentSet)
        {
            for (Block nblock : Utility.getSurroundingBlocks(currentblock, true, true, true))
            {
                if (!(initialMaterial == nblock.getType() && initialData == nblock.getData()))
                    continue;
                if (nextUpdate.contains(nblock))
                    continue;
                nextUpdate.add(nblock);
            }
        }
        currentSet = nextUpdate;
    }

    public boolean isDone()
    {
        return currentSet.size() == 0 ||
                blocksModified >= maxModification ||
                outOfBlocks;
    }
}

class TerramorpherJob implements Job
{
    public Block initialBlock;
    public Material initialMaterial;
    public Orientation orientation;
    public int size;

    private ArrayList<Block> currentSet;
    private Block minimumConstraint;
    private Block maximumConstraint;

    public TerramorpherJob(Block initialBlock, BlockFace orientFace, int size)
    {
        this.initialBlock = initialBlock;
        this.initialMaterial = initialBlock.getType();
        this.orientation = Orientation.fromBlockFace(orientFace);
        this.size = size;

        this.currentSet = new ArrayList<>();
        currentSet.add(initialBlock);
        Vector offsetVector = orientation.inverseVector().multiply(size);
        minimumConstraint = initialBlock.getLocation().subtract(offsetVector).getBlock();
        maximumConstraint = initialBlock.getLocation().add(offsetVector).getBlock();
    }

    public void update()
    {
        for (Block b : currentSet)
        {
            b.breakNaturally(new ItemStack(Material.DIAMOND_SPADE));
            b.getWorld().playEffect(b.getLocation(), Effect.TILE_BREAK, new MaterialData(b.getType()));
        }
        ArrayList<Block> nextUpdate = new ArrayList<>();
        for (Block currentblock : currentSet)
        {
            for (Block nblock : Utility.getSurroundingBlocks(currentblock, true, false, false))
            {
                if (initialMaterial != nblock.getType())
                    continue;
                if (nextUpdate.contains(nblock))
                    continue;
                if (!Utility.blockInBoundsInclusive(minimumConstraint, maximumConstraint, nblock))
                    continue;
                nextUpdate.add(nblock);
            }
        }
        currentSet = nextUpdate;
    }

    public boolean isDone()
    {
        return currentSet.size() == 0;
    }

    enum Orientation
    {
        NORTH_SOUTH (0, 0, 1),
        EAST_WEST (1, 0, 0),
        UP_DOWN (0, 1, 0);

        double x;
        double y;
        double z;

        Orientation(double x, double y, double z)
        {
            this.x = x;
            this.y = y;
            this.z = z;
        }

        public Vector getVector()
        {
            return new Vector(x, y, z);
        }

        public Vector inverseVector()
        {
            return new Vector(1, 1, 1).subtract(getVector());
        }

        public static Orientation fromBlockFace(BlockFace face)
        {
            if (face == BlockFace.NORTH || face == BlockFace.SOUTH)
                return NORTH_SOUTH;
            else if (face == BlockFace.EAST || face == BlockFace.WEST)
                return EAST_WEST;
            else if (face == BlockFace.UP || face == BlockFace.DOWN)
                return UP_DOWN;
            return null;
        }
    }
}