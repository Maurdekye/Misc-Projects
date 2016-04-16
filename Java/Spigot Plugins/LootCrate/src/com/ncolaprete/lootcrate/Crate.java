package com.ncolaprete.lootcrate;

import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.entity.Player;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.InventoryHolder;
import org.bukkit.inventory.ItemStack;
import org.bukkit.material.Colorable;

import java.util.ArrayList;
import java.util.List;
import java.util.ListIterator;
import java.util.stream.Collectors;

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
    List<Reward> contents;

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
        int prizeIndex = Utility.randomWeightedIndex(contents.stream().map(c -> c.rewardChance).collect(Collectors.toList()));
        if (prizeIndex == -1)
        {
            rewardee.getServer().getLogger().info("Probability Error in chest layout " + type + "!");
            return;
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
        double totalProbability = 0;
        for (Reward r : contents)
            totalProbability += r.rewardChance;
        List<ItemStack> itemsToShow = new ArrayList<>();
        for (Reward r : contents)
        {
            double chance = (r.rewardChance / totalProbability) * 10000;
            chance = Math.round(chance) / 100;
            ItemStack displayItem = r.item.getVisualisation(plugin, ply, r.amount, chestblock);
            Utility.addLoreLine(displayItem, ChatColor.WHITE + "%" + chance + " chance");
            itemsToShow.add(displayItem);
        }
        int invSize = itemsToShow.size() - (itemsToShow.size()%9) + 9;
        Inventory display = Bukkit.createInventory((InventoryHolder) chestblock.getState(), invSize, getPrintname(true));
        for (int i=0;i<invSize;i++)
        {
            ItemStack pane = Utility.setName(Material.STAINED_GLASS_PANE, " ");
            pane.setDurability((short)15);
            display.setItem(i, pane);
        }
        ListIterator<ItemStack> iter = itemsToShow.listIterator();
        for (int i=0;i<invSize;i++)
        {
            if (!iter.hasNext())
                break;
            display.setItem(i, iter.next());
        }
        return display;
    }

    public String toString()
    {
        return type.toLowerCase();
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
        ItemStack crateDrop = Utility.setName(Material.CHEST, getPrintname(true));
        if (keyRequired != null)
            Utility.addLoreLine(crateDrop, ChatColor.RESET + "" + ChatColor.GRAY + "Requires a " + keyRequired.displayname + ChatColor.RESET + ChatColor.GRAY + " to unlock");
        else
            Utility.addLoreLine(crateDrop, ChatColor.RESET + "" + ChatColor.GRAY + "Does not require a key");
        Utility.addLoreLine(crateDrop, getLoreTag());
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
    public double buyprice;
    public List<String> lore;

    public CrateKey(String type, Material material, String displayname, double buyprice, List<String> lore)
    {
        this.type = type;
        this.material = material;
        this.displayname = displayname;
        this.buyprice = buyprice;
        this.lore = lore;
    }

    public ItemStack getKey(boolean isDisplayKey)
    {
        ItemStack key = Utility.setName(material, displayname);
        if (isDisplayKey)
            return key;
        for (String line : lore)
            Utility.addLoreLine(key, line);
        key = Utility.addLoreLine(key, getLoreTag());
        return key;
    }

    public String getLoreTag()
    {
        return ChatColor.BLACK + type;
    }

    public String toString()
    {
        return type.toLowerCase();
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