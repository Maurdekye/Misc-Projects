package com.ncolaprete.lootcrate;

import net.ess3.api.Economy;
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
import org.bukkit.entity.Entity;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityDamageEvent;
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
import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

public class LootCrate extends JavaPlugin implements Listener, CommandExecutor{

    public ArrayList<CrateKey> crateKeys;
    public ArrayList<CrateLayout> crateLayouts;
    public ArrayList<Crate> cratePositions;
    public HashMap<UUID, Long> tempCreativeTimestamps;
    public ArrayList<Job> activeJobs;

    public CustomConfig crateKeyConfig;
    public CustomConfig crateLayoutConfig;
    public CustomConfig cratePositionConfig;
    public CustomConfig tempCreativeTimestampConfig;
    public CustomConfig optionsConfig;

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

    public double WandOfLeapingPower;

    // Overridden Methods

    public void onEnable()
    {
        // register listeners / repeating events
        getServer().getPluginManager().registerEvents(this, this);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::updateJobs, 0, 1);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::checkAllPlayersForSpecialItems, 0, 20);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::checkForCreativeTimeUp, 0, 1200);
        getServer().getScheduler().scheduleSyncRepeatingTask(this, this::checkForInvalidCrateLocations, 0, 36000);

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
            double buyprice;
            try {
                buyprice = Double.parseDouble(keysection.getString("price", "0"));
            } catch (Exception e) {
                csend.sendMessage(ChatColor.RED + "Error! '" + keysection.getString("price") + "' is not a number!");
                continue;
            }
            String lorestring = keysection.getString("description", "");
            lorestring = ChatColor.translateAlternateColorCodes('?', lorestring);
            List<String> lore = Arrays.asList(lorestring.split("\\\\n"));
            crateKeys.add(new CrateKey(type, material, name, buyprice, lore));
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
            double spawnChance = cratesection.getDouble("spawn_chance", 0);
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

        ConfigurationSection wandofleapingCfg = optionsConfig.getConfig().getConfigurationSection("wand_of_leaping");
        WandOfLeapingPower = wandofleapingCfg.getInt("power");

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

        if (command.getName().equalsIgnoreCase("buykey"))
        {
            String genericErrorMessage = ChatColor.RED + "Cannot buy any keys at this time.";

            // Check if sender is a player
            if (ply == null)
            {
                sender.sendMessage(ChatColor.RED + "You must be a player to use this command.");
                return true;
            }

            // Check if there are any buyable crate keys available
            if (crateKeys.stream().filter(k -> k.buyprice > 0).collect(Collectors.toList()).size() == 0)
            {
                sender.sendMessage(genericErrorMessage);
                return true;
            }

            // Check if enough arguments were provided
            if (args.length == 0)
            {
                ply.sendMessage("Keys available for purchase: ");
                for (CrateKey k : crateKeys.stream().filter(kf -> kf.buyprice > 0).collect(Collectors.toList()))
                    ply.sendMessage("|    " + k.type.toLowerCase() + ": $" + k.buyprice);
                return false;
            }

            // Find key to buy
            Optional<CrateKey> maybeKey = crateKeys.stream().filter(k -> k.type.equalsIgnoreCase(args[0]) && k.buyprice >= 0).findFirst();
            if (!maybeKey.isPresent())
            {
                ply.sendMessage(ChatColor.RED + "No crate key '" + args[0] + "'.");
                return true;
            }
            CrateKey key = maybeKey.get();

            // Find amount of keys to buy
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

            // Check if key is buyable
            if (key.buyprice == 0)
            {
                ply.sendMessage(ChatColor.RED + "Key '" + key.type + "' Is not available for purchase.");
                return true;
            }

            // Check if player has enough money
            if (Utility.getBalance(ply).compareTo(new BigDecimal(key.buyprice)) == -1)
            {
                if (amount == 1)
                    ply.sendMessage(ChatColor.RED + "You do not have enough money; this key costs $" + key.buyprice + ".");
                else
                    ply.sendMessage(ChatColor.RED + "You do not have enough money; these keys cost $" + (key.buyprice*amount) + " total.");
                return true;
            }

            // Subtract balance and give key
            if (amount == 1)
                ply.sendMessage("You bought a " + key.displayname + ChatColor.RESET + "!");
            else
                ply.sendMessage("You bought " + amount + " " + key.displayname + "s" + ChatColor.RESET + "!");
            Utility.modifyBalance(ply, new BigDecimal(key.buyprice).negate());
            getServer().dispatchCommand(csend, "givekey " + key.type + " " + amount + " " + ply.getName());
        }

        // givekey
        else if (command.getName().equalsIgnoreCase("givekey"))
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

    private void checkForInvalidCrateLocations()
    {
        ArrayList<Integer> toRemove = new ArrayList<>();
        for (int i=0;i<cratePositions.size();i++)
            if (cratePositions.get(i).location.getType() != Material.CHEST)
                toRemove.add(i);
        toRemove.stream().forEach(this::removeCrate);
    }

    // Event Handlers

    @SuppressWarnings("deprecation")
    @EventHandler
    public void playerInteract(PlayerInteractEvent ev)
    {
        Player ply = ev.getPlayer();

        // manage opening of crates
        if (ev.getAction() == Action.RIGHT_CLICK_BLOCK && !ply.isSneaking())
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
            if (!ply.hasPermission("lootcrate.opencrate"))
            {
                ply.sendMessage(ChatColor.RED + "You do not have permission to open crates.");
                ev.setCancelled(true);
                return;
            }
            ItemStack handItem = ply.getInventory().getItemInMainHand();
            if (crateToOpen.isKeyValid(handItem))
            {
                if (!crateToOpen.layout.isFree())
                {
                    if (handItem.getAmount() > 1)
                        handItem.setAmount(handItem.getAmount() - 1);
                    else
                        ply.getInventory().remove(handItem);
                }
                crateToOpen.unlockAndGivePrize(this, ply);
                removeCrate(crateToOpen);
            }
            else
            {
                ev.setCancelled(true);
                ply.openInventory(crateToOpen.showContents(this, ply));
            }
        }

        // Transmogrifier
        if (ev.getAction() == Action.LEFT_CLICK_BLOCK &&
                Prize.itemIsPrize(ply.getInventory().getItemInMainHand(), Prize.TRANSMOGRIFIER))
        {
            ItemStack offhandItem = ply.getInventory().getItemInOffHand();
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
            activeJobs.add(new TransmogrificationJob(ev.getClickedBlock(), ply, TransmogiphySpeed, MaxBlocksPerTransmogrophy));
        }

        // Wand of Leaping
        if ((ev.getAction() == Action.LEFT_CLICK_BLOCK || ev.getAction() == Action.LEFT_CLICK_AIR) &&
                Prize.itemIsPrize(ply.getInventory().getItemInMainHand(), Prize.WAND_OF_LEAPING))
        {
            if (!Utility.isOnGround(ply))
                return;
            Vector direction = ply.getLocation().getDirection().normalize();
            direction = direction.multiply(WandOfLeapingPower);
            ply.setVelocity(direction);
            ply.playSound(ply.getLocation(), Sound.ENTITY_PLAYER_ATTACK_SWEEP, 1, 1);
            if (ply.getGameMode() != GameMode.CREATIVE)
                Utility.reduceDurability(ply, ply.getInventory().getItemInMainHand(), 1);
        }
    }

    @EventHandler
    public void blockBreak(BlockBreakEvent ev)
    {
        // Treefeller Chainsaw
        if (Prize.itemIsPrize(ev.getPlayer().getInventory().getItemInMainHand(), Prize.TREEFELLER_CHAINSAW))
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
        if (Prize.itemIsPrize(ev.getPlayer().getInventory().getItemInMainHand(), Prize.TERRAMORPHER))
        {
            if (!Utility.isCorrectTool(Material.DIAMOND_SPADE, ev.getBlock().getType()))
                return;
            activeJobs.add(new TerramorpherJob(ev.getBlock(), Utility.getBlockFaceIsLookingAt(ev.getPlayer()), TerramorpherSize));
        }

        // Giga Drill Breaker
        if (Prize.itemIsPrize(ev.getPlayer().getInventory().getItemInMainHand(), Prize.GIGA_DRILL_BREAKER))
        {
            if (!Utility.isCorrectTool(Material.DIAMOND_PICKAXE, ev.getBlock().getType()))
                return;
            activeJobs.add(new TerramorpherJob(ev.getBlock(), Utility.getBlockFaceIsLookingAt(ev.getPlayer()), GigaDrillBreakerSize));
        }

        // Transmogriphier
        if (Prize.itemIsPrize(ev.getPlayer().getInventory().getItemInMainHand(), Prize.TRANSMOGRIFIER) &&
                ev.getPlayer().getGameMode() == GameMode.CREATIVE)
        {
            ev.setCancelled(true);
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
            if (Prize.itemIsPrize(ply.getInventory().getItemInMainHand(), Prize.HYPERSHOT_LONGBOW))
            {
                ev.getEntity().setVelocity(ev.getEntity().getVelocity().multiply(4));
            }
        }
    }

    @EventHandler
    public void entityDamage(EntityDamageEvent ev)
    {
        if (!(ev.getEntity() instanceof Player))
            return;
        Player ply = (Player) ev.getEntity();

        // Wand of Leaping
        if (ev.getCause() == EntityDamageEvent.DamageCause.FALL &&
                Prize.itemIsPrize(ply.getInventory().getItemInMainHand(), Prize.WAND_OF_LEAPING))
        {
            ev.setCancelled(true);
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
        // get layout to use
        ArrayList<Double> weights = new ArrayList<>();
        for (CrateLayout l : crateLayouts)
            weights.add(l.spawnChance);
        int index = Utility.randomWeightedIndex(weights);
        if (index == -1)
        {
            csend.sendMessage(ChatColor.RED + "Error: Attempted to place crate when none have any chance to drop. Please disable random crate drops in lootcrate_options.yml if this is intended.");
            return;
        }
        CrateLayout layout = crateLayouts.get(Utility.randomWeightedIndex(weights));

        // get location to use
        Location droplocation;
        Block newChest;
        do {
            droplocation = center.add(Utility.randomInsideUnitCircle().multiply(radius));
            newChest = Utility.getHighestSolidBlock(center.getWorld(), droplocation.getBlockX(), droplocation.getBlockZ());
        } while (newChest.getLocation().getY() >= droplocation.getWorld().getMaxHeight());

        // broadcast crate position
        if (layout.shouldBroadcast)
        {
            getServer().broadcastMessage("A " + layout.printname + ChatColor.RESET + " has dropped at " + ChatColor.GOLD + newChest.getX() + ", " + newChest.getZ() + ChatColor.RESET + "!");
        }
        csend.sendMessage(layout.printname + ChatColor.RESET + " spawned at " + Utility.formatVector(newChest.getLocation().toVector()));

        // drop crate
        addCrate(new Crate(newChest.getRelative(BlockFace.UP), layout));
    }

    private void checkPlayerForSpecialItem(Player ply)
    {
        ItemStack mainHandItem = ply.getInventory().getItemInMainHand();
        ItemStack offHandItem = ply.getInventory().getItemInOffHand();

        // Frostspark Cleats
        if (Prize.itemIsPrize(ply.getInventory().getBoots(), Prize.FROSTSPARK_CLEATS))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.SPEED, 40, 2), true);
            ply.addPotionEffect(new PotionEffect(PotionEffectType.JUMP, 40, 1), true);
        }

        // Lucky Trousers
        if (Prize.itemIsPrize(ply.getInventory().getLeggings(), Prize.LUCKY_TROUSERS))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.LUCK, 40, 2), true);
        }

        // Knackerbreaker Chesterplate
        if (Prize.itemIsPrize(ply.getInventory().getChestplate(), Prize.KNACKERBREAKER_CHESTERPLATE))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.ABSORPTION, 40, 0), true);
        }

        // Hydrodyne Helmet
        if (Prize.itemIsPrize(ply.getInventory().getHelmet(), Prize.HYDRODYNE_HELMET))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.NIGHT_VISION, 250, 0), true);
            ply.addPotionEffect(new PotionEffect(PotionEffectType.WATER_BREATHING, 40, 0), true);
        }

        // Giga Drill Breaker
        if (Prize.itemIsPrize(mainHandItem, Prize.GIGA_DRILL_BREAKER))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.FAST_DIGGING, 25, 3), true);
        }

        // Unyielding Battersea
        if ((Prize.itemIsPrize(offHandItem, Prize.UNYIELDING_BATTERSEA) ||
                Prize.itemIsPrize(mainHandItem, Prize.UNYIELDING_BATTERSEA)) &&
                ply.isBlocking())
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.DAMAGE_RESISTANCE, 40, 0), true);
            ply.addPotionEffect(new PotionEffect(PotionEffectType.FIRE_RESISTANCE, 40, 0), true);
        }

        // Veilstrike Shortbow
        if (Prize.itemIsPrize(mainHandItem, Prize.VEILSTRIKE_SHORTBOW))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.INVISIBILITY, 25, 0), true);
        }

        // Heaven's Blade
        if (Prize.itemIsPrize(mainHandItem, Prize.HEAVENS_BLADE))
        {
            ply.addPotionEffect(new PotionEffect(PotionEffectType.INCREASE_DAMAGE, 25, 4), true);
        }
    }
}
