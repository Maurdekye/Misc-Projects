package com.ncolaprete.lootcrate;

import net.minecraft.server.v1_9_R1.TileEntityChest;
import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.block.Chest;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.craftbukkit.v1_9_R1.block.CraftChest;
import org.bukkit.entity.EntityType;
import org.bukkit.entity.Firework;
import org.bukkit.entity.Ocelot;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.FireworkMeta;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;
import org.bukkit.util.BlockIterator;
import org.bukkit.util.Vector;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class LootCrate extends JavaPlugin implements Listener, CommandExecutor{

    ArrayList<Crate> allCrates;
    ArrayList<CrateLayout> crateLayouts;

    public static final String LockedTag = ChatColor.RED + " [Locked]";
    public static final String UnlockedTag = ChatColor.YELLOW + " [Unlocked]";

    // Overridden Methods

    public void onEnable()
    {
        getServer().getPluginManager().registerEvents(this, this);

        allCrates = new ArrayList<>();
        crateLayouts = new ArrayList<>();
        ArrayList<Reward> mysticCrateRewardList = new ArrayList<>();
        mysticCrateRewardList.add(new Reward(Prize.INVINCIBILITY, 600, 30));
        mysticCrateRewardList.add(new Reward(Prize.MONEY, 1000, 30));
        mysticCrateRewardList.add(new Reward(Prize.IRON_BARS, 64, 15));
        mysticCrateRewardList.add(new Reward(Prize.MONEY, 500, 10));
        mysticCrateRewardList.add(new Reward(Prize.MONEY, 2000, 5));
        mysticCrateRewardList.add(new Reward(Prize.DIAMONDS, 16, 4));
        mysticCrateRewardList.add(new Reward(Prize.DIAMONDS, 64, 3));
        mysticCrateRewardList.add(new Reward(Prize.CRATE_KEY, 1, 2));
        mysticCrateRewardList.add(new Reward(Prize.MYSTIC_KEY, 1, 1));
        crateLayouts.add(new CrateLayout(
                ChatColor.LIGHT_PURPLE + "Mystic Crate",
                "mystic_crate",
                Prize.MYSTIC_KEY,
                mysticCrateRewardList
        ));
        ArrayList<Reward> commonCrateRewardList = new ArrayList<>();
        commonCrateRewardList.add(new Reward(Prize.IRON_BARS, 16, 30));
        commonCrateRewardList.add(new Reward(Prize.INDIVIDUAL_NUGGETS, 8, 30));
        commonCrateRewardList.add(new Reward(Prize.MONEY, 100, 15));
        commonCrateRewardList.add(new Reward(Prize.MONEY, 500, 10));
        commonCrateRewardList.add(new Reward(Prize.MONEY, 2000, 5));
        commonCrateRewardList.add(new Reward(Prize.DIAMONDS, 1, 4));
        commonCrateRewardList.add(new Reward(Prize.DIAMONDS, 8, 3));
        commonCrateRewardList.add(new Reward(Prize.CRATE_KEY, 1, 2));
        commonCrateRewardList.add(new Reward(Prize.MYSTIC_KEY, 1, 1));
        crateLayouts.add(new CrateLayout(
                ChatColor.GREEN + "Common Crate",
                "common_crate",
                Prize.CRATE_KEY,
                commonCrateRewardList
        ));
    }

    public boolean onCommand(CommandSender sender, Command command, String label, String[] args)
    {
        Player ply = sender instanceof Player ? (Player) sender : null;

        // givecrate
        if (command.getName().equalsIgnoreCase("givecrate"))
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

            // give crate
            target.getInventory().addItem(Utility.setName(new ItemStack(Material.CHEST), layout.getPrintname(true)));
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
            allCrates.add(new Crate(location, layout));
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
            type.giveReward(ply, amount, chestBlock);
        }
        return true;
    }

    // Event Handlers

    @EventHandler
    public void playerInteract(PlayerInteractEvent ev)
    {
        if (ev.getAction() == Action.RIGHT_CLICK_BLOCK)
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
                crateToOpen.unlockAndGivePrize(ev.getPlayer());
                allCrates.remove(crateToOpen);
            }
            else
            {
                ev.setCancelled(true);
                ev.getPlayer().openInventory(crateToOpen.showContents(ev.getPlayer()));
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
                ItemStack crateDrop = Utility.setName(new ItemStack(Material.CHEST), allCrates.get(i).layout.getPrintname(true));
                ev.getBlock().setType(Material.AIR);
                ev.getBlock().getWorld().dropItemNaturally(ev.getBlock().getLocation().add(0.5, 0.5, 0.5), crateDrop);
                allCrates.remove(i);
                break;
            }
        }
    }

    @EventHandler
    public void blockPlace(BlockPlaceEvent ev)
    {
        String itemName = Utility.getName(ev.getItemInHand());
        for (CrateLayout l : crateLayouts)
        {
            if (l.getPrintname(true).equals(itemName))
            {
                allCrates.add(new Crate(ev.getBlock(), l));
                break;
            }
        }
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

}

class Utility
{
    public static boolean itemHasLoreLine(ItemStack item, String line)
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
    public static ItemStack addLoreLine(ItemStack item, String line)
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

    public static ItemStack setName(ItemStack item, String name)
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

    public static int randomInt(int start, int end)
    {
        return (int)(Math.random() * (end - start)) + start;
    }

    public static void setChestInventoryName(Block chestblock, String name)
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

    public void unlockAndGivePrize(Player rewardee)
    {
        layout.givePrize(rewardee, location);
        Utility.setChestInventoryName(location, layout.getPrintname(false));
    }

    public Inventory showContents(Player ply)
    {
        return layout.showContents(ply, location);
    }

    public boolean isKeyValid(ItemStack key)
    {
        return layout.isKeyValid(key);
    }
}

class CrateLayout
{
    public String printname;
    public String type;
    public Prize keyRequired;
    public ArrayList<Reward> contents;

    public CrateLayout(String printname, String type, Prize keyRequired, ArrayList<Reward> contents)
    {
        this.printname = printname;
        this.type = type;
        this.keyRequired = keyRequired;
        this.contents = contents;
    }

    public void givePrize(Player rewardee, Block location)
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
        chosen.item.giveReward(rewardee, chosen.amount, location);
        rewardee.playSound(rewardee.getLocation(), Sound.ENTITY_PLAYER_LEVELUP, 1, 1);
    }

    public Inventory showContents(Player ply, Block chestblock)
    {
        Inventory display = Bukkit.createInventory(null, contents.size(), getPrintname(true));
        for (Reward r : contents)
        {
            ItemStack displayItem =  r.item.getVisualisation(ply, r.amount, chestblock);
            displayItem = Utility.addLoreLine(displayItem, ChatColor.WHITE + "%" + r.rewardChance + " chance");
            display.addItem(displayItem);
        }
        return display;
    }

    public boolean isKeyValid(ItemStack key)
    {
        return Utility.itemHasLoreLine(key, keyRequired.getLoreTag());
    }

    public String getPrintname(boolean isLocked)
    {
        if (isLocked)
            return printname + LootCrate.LockedTag;
        return printname + LootCrate.UnlockedTag;
    }
}

class Reward
{
    public Prize item;
    public int amount;
    public double rewardChance;

    public Reward(Prize item, int amount, double rewardChance)
    {
        this.item = item;
        this.amount = amount;
        this.rewardChance = rewardChance;
    }
}

enum Prize
{
    MYSTIC_KEY (params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.LIGHT_PURPLE + "Mystic Key");
        keyMeta.setLore(Arrays.asList("A mystical key of unknown origin. Surely very rare.", ChatColor.BLACK + "mystic_key"));
        key.setItemMeta(keyMeta);
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Mystic Keys!");
        else
            params.rewardee.sendMessage("You got a Mystic Key!");
        return Collections.singletonList(key);
    }, params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.LIGHT_PURPLE + "Mystic Key");
        keyMeta.setLore(Collections.singletonList("A mystical key of unknown origin. Surely very rare."));
        key.setItemMeta(keyMeta);
        return key;
    }),

    CRATE_KEY (params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.AQUA + "Crate Key");
        keyMeta.setLore(Arrays.asList("A normal crate key.", ChatColor.BLACK + "crate_key"));
        key.setItemMeta(keyMeta);
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Crate Keys!");
        else
            params.rewardee.sendMessage("You got a Crate Key!");
        return Collections.singletonList(key);
    }, params -> {
        ItemStack key = new ItemStack(Material.TRIPWIRE_HOOK, params.amountToGive);
        ItemMeta keyMeta = key.getItemMeta();
        keyMeta.setDisplayName(ChatColor.AQUA + "Crate Key");
        keyMeta.setLore(Collections.singletonList("A normal crate key."));
        key.setItemMeta(keyMeta);
        return key;
    }),

    DIAMONDS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Diamonds!");
        else
            params.rewardee.sendMessage("You got a Diamond!");
        return Collections.singletonList(new ItemStack(Material.DIAMOND, params.amountToGive));
    }, params -> {
        ItemStack item = new ItemStack(Material.DIAMOND, 1);
        ItemMeta meta = item.getItemMeta();
        if (params.amountToGive > 1)
            meta.setDisplayName(ChatColor.DARK_AQUA + "" + params.amountToGive + " Diamonds");
        else
            meta.setDisplayName(ChatColor.DARK_AQUA + "1 Diamond");
        item.setItemMeta(meta);
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
        ItemMeta meta = item.getItemMeta();
        if (params.amountToGive > 1)
            meta.setDisplayName(ChatColor.DARK_AQUA + "" + params.amountToGive + " Iron Ingots");
        else
            meta.setDisplayName(ChatColor.DARK_AQUA + "1 Iron Ingot");
        item.setItemMeta(meta);
        return item;
    }),

    MONEY (params -> {
        params.rewardee.getServer().dispatchCommand(params.rewardee.getServer().getConsoleSender(), "eco give " + params.rewardee.getName() + " " + params.amountToGive);
        params.rewardee.sendMessage("You got $" + params.amountToGive + "!");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.GOLD_INGOT, 1);
        ItemMeta meta = item.getItemMeta();
        meta.setDisplayName(ChatColor.DARK_AQUA + "$" + params.amountToGive);
        item.setItemMeta(meta);
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
        item.setItemMeta(meta);
        return item;
    }),

    INDIVIDUAL_NUGGETS (params -> {
        ArrayList<ItemStack> nuggets = new ArrayList<>();
        for (int i=0;i<params.amountToGive;i++)
            nuggets.add(new ItemStack(Material.GOLD_NUGGET, 1));
        params.rewardee.sendMessage("You got " + params.amountToGive + " individually placed gold nuggets!");
        return nuggets;
    }, params -> {
        ItemStack item = new ItemStack(Material.GOLD_NUGGET);
        ItemMeta meta = item.getItemMeta();
        if (params.amountToGive > 1)
            meta.setDisplayName(ChatColor.DARK_AQUA + "" + params.amountToGive + " Individually Placed Gold Nuggets");
        else
            meta.setDisplayName(ChatColor.DARK_AQUA + "1 Gold Nugget");
        item.setItemMeta(meta);
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
        params.rewardee.sendMessage(ChatColor.AQUA + "You got a fireworks show! Yaayyy!!");
        return null;
    }, params -> {
        ItemStack item = new ItemStack(Material.FIREWORK);
        ItemMeta meta = item.getItemMeta();
        meta.setDisplayName(ChatColor.GOLD + "A Fireworks Show");
        item.setItemMeta(meta);
        return item;
    });

    private PrizeAction action;
    private PrizeVisual visualisation;
    Prize(PrizeAction action, PrizeVisual visualisation)
    {
        this.action = action;
        this.visualisation = visualisation;
    }

    public void giveReward(Player rewardee, int amount, Block chestBlock)
    {
        if (chestBlock.getType() != Material.CHEST)
            return;
        List<ItemStack> rewardItemsRaw = action.enactReward(new RewardActionParameter(rewardee, amount, chestBlock));
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

    public ItemStack getVisualisation(Player rewardee, int amount, Block chestBlock)
    {
        return visualisation.getVisualisation(new RewardActionParameter(rewardee, amount, chestBlock));
    }

    public String getLoreTag()
    {
        return ChatColor.BLACK + toString().toLowerCase();
    }

    public class RewardActionParameter
    {
        public Player rewardee;
        public int amountToGive;
        public Block chestBlock;

        public RewardActionParameter(Player rewardee, int amountToGive, Block chestBlock)
        {
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