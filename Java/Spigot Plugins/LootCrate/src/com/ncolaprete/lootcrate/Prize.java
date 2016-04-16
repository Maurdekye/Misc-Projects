package com.ncolaprete.lootcrate;

import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.block.Chest;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.EntityType;
import org.bukkit.entity.Firework;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.FireworkMeta;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.inventory.meta.PotionMeta;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;
import org.bukkit.util.Vector;

import java.math.BigDecimal;
import java.util.*;

enum Prize implements LoreTaggable
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
    }, params -> Utility.setName(Material.COMMAND, ChatColor.UNDERLINE + "" + ChatColor.BOLD + "" + "The Ultimate Reward")),

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
    }, params -> Utility.setName(Material.DIAMOND_BOOTS, ChatColor.YELLOW + "Frostspark Cleats")),

    WATERGLIDE_BOOTS (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_BOOTS);
        item = Utility.setName(item, ChatColor.AQUA + "Waterglide Boots");
        item.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 1);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.DEPTH_STRIDER, 3);
        params.rewardee.sendMessage(ChatColor.AQUA + "You got the Waterglide Boots!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.DIAMOND_BOOTS, ChatColor.AQUA + "Waterglide Boots")),

    LUCKY_TROUSERS (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_LEGGINGS);
        item = Utility.setName(item, ChatColor.GREEN + "Lucky Trousers");
        item = Utility.addLoreLine(item, ChatColor.RESET + "The trousers grant increased luck");
        item.addEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 1);
        item.addEnchantment(Enchantment.DURABILITY, 1);
        params.rewardee.sendMessage(ChatColor.GREEN + "You got the Lucky Trousers!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.DIAMOND_LEGGINGS, ChatColor.GREEN + "Lucky Trousers")),

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
    }, params -> Utility.setName(Material.DIAMOND_CHESTPLATE, ChatColor.GOLD + "Knackerbreaker Chesterplate")),

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
    }, params -> Utility.setName(Material.DIAMOND_HELMET, ChatColor.BLUE + "Hydrodyne Helmet")),

    // Tools

    ANTIMATTER_DEMATERIALIZER (params -> {
        ItemStack item = new ItemStack(Material.GOLD_PICKAXE);
        Utility.setName(item, ChatColor.GRAY + "" + ChatColor.ITALIC + ChatColor.BOLD + "Antimatter Dematerializer");
        Utility.addLoreLine(item, ChatColor.GRAY + "" + ChatColor.ITALIC + "It can only be used once.");
        Utility.addLoreLine(item, ChatColor.DARK_GRAY + "" + ChatColor.ITALIC + "But what is done, cannot be undone...");
        params.rewardee.sendMessage(ChatColor.GRAY + "You have been burdened with the Antimatter Dematerializer.");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.GOLD_PICKAXE, ChatColor.GRAY + "" + ChatColor.ITALIC + ChatColor.BOLD + "Antimatter Dematerializer")),

    WAND_OF_LEAPING (params -> {
        ItemStack item = new ItemStack(Material.GOLD_HOE);
        Utility.setName(item, ChatColor.LIGHT_PURPLE + "" + ChatColor.ITALIC + "Wand of Leaping");
        Utility.addLoreLine(item, ChatColor.RESET + "Allows the user to leap great distances and avoid fall damage");
        item.addEnchantment(Enchantment.DURABILITY, 1);
        params.rewardee.sendMessage(ChatColor.LIGHT_PURPLE + "You got the " + ChatColor.ITALIC + "Wand of Leaping!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.GOLD_HOE, ChatColor.LIGHT_PURPLE + "" + ChatColor.ITALIC + "Wand of Leaping")),

    TERRAMORPHER (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_SPADE);
        Utility.setName(item, ChatColor.GREEN + "" + ChatColor.BOLD + "Terramorpher");
        Utility.addLoreLine(item, ChatColor.RESET + "The terramorpher digs multiple blocks at once");
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.DIG_SPEED, 3);
        params.rewardee.sendMessage(ChatColor.GREEN + "You got the " + ChatColor.BOLD + "Terramorpher!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.DIAMOND_SPADE, ChatColor.GREEN + "" + ChatColor.BOLD + "Terramorpher")),

    TRANSMOGRIFIER (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_HOE);
        Utility.setName(item, ChatColor.GOLD + "" + ChatColor.BOLD + "Transmogrifier");
        Utility.addLoreLine(item, ChatColor.RESET + "The transmogrifier will swap blocks with those in your offhand with a left click");
        Utility.addLoreLine(item, ChatColor.RESET + "...you can also till soil with it.");
        item.addEnchantment(Enchantment.DURABILITY, 2);
        params.rewardee.sendMessage(ChatColor.GOLD + "You got the " + ChatColor.BOLD + "Transmogrifier!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.DIAMOND_HOE, ChatColor.GOLD + "" + ChatColor.BOLD + "Transmogrifier")),

    TREEFELLER_CHAINSAW (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_AXE);
        Utility.setName(item, ChatColor.DARK_GREEN + "" + ChatColor.ITALIC + "Treefeller Chainsaw");
        Utility.addLoreLine(item, "Let gravity do the work for you!");
        Utility.addLoreLine(item, ChatColor.RESET + "The chainsaw fells entire trees with a single blow");
        item.addEnchantment(Enchantment.DURABILITY, 2);
        params.rewardee.sendMessage(ChatColor.DARK_GREEN + "You got the Treefeller Chainsaw!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.DIAMOND_AXE, ChatColor.DARK_GREEN + "" + ChatColor.ITALIC + "Treefeller Chainsaw")),

    GIGA_DRILL_BREAKER (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_PICKAXE);
        Utility.setName(item, ChatColor.AQUA + "" + ChatColor.BOLD + "Giga Drill Breaker");
        Utility.addLoreLine(item, ChatColor.AQUA + "Bust through the heavens with your Drill!");
        item.addEnchantment(Enchantment.DIG_SPEED, 5);
        item.addEnchantment(Enchantment.DURABILITY, 3);
        params.rewardee.sendMessage(ChatColor.AQUA + "You got the Giga Drill Breaker; thrust through the heavens with your spirit!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.DIAMOND_PICKAXE, ChatColor.AQUA + "" + ChatColor.BOLD + "Giga Drill Breaker")),

    UNYIELDING_BATTERSEA (params -> {
        ItemStack item = new ItemStack(Material.SHIELD);
        Utility.setName(item, ChatColor.YELLOW + "Unyielding Battersea");
        Utility.addLoreLine(item, "An olden shield used by unending legions");
        Utility.addLoreLine(item, ChatColor.RESET + "The battersea grants increase resistances while equipped");
        item.addEnchantment(Enchantment.DURABILITY, 3);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "You got the Unyielding Battersea!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.SHIELD, ChatColor.YELLOW + "Unyielding Battersea")),

    HYPERSHOT_LONGBOW (params -> {
        ItemStack item = new ItemStack(Material.BOW);
        Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.ITALIC + "Hypershot Longbow");
        Utility.addLoreLine(item, "An ancient, powerful bow used by a skilled marksman");
        Utility.addLoreLine(item, ChatColor.RESET + "The bow grants immense arrow speed and power");
        item.addEnchantment(Enchantment.ARROW_DAMAGE, 5);
        item.addEnchantment(Enchantment.ARROW_KNOCKBACK, 2);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "" + ChatColor.ITALIC + "You got the Hypershot Longbow!");
        return Arrays.asList(item, new ItemStack(Material.ARROW, params.amountToGive));
    }, params -> Utility.setName(Material.BOW, ChatColor.YELLOW + "" + ChatColor.ITALIC + "Hypershot Longbow")),

    VEILSTRIKE_SHORTBOW (params -> {
        ItemStack item = new ItemStack(Material.BOW);
        Utility.setName(item, ChatColor.BLUE + "" + ChatColor.ITALIC + "Veilstrike Shortbow");
        Utility.addLoreLine(item, "An ancient, mystical bow used by an unseen asassin");
        Utility.addLoreLine(item, ChatColor.RESET + "The bow grants invisibility while held");
        item.addEnchantment(Enchantment.ARROW_DAMAGE, 3);
        item.addEnchantment(Enchantment.DURABILITY, 2);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.BLUE + "" + ChatColor.ITALIC + "You got the Veilstrike Shortbow!");
        return Arrays.asList(item, new ItemStack(Material.ARROW, params.amountToGive));
    }, params -> Utility.setName(Material.BOW, ChatColor.BLUE + "" + ChatColor.ITALIC + "Veilstrike Shortbow")),

    HEAVENS_BLADE (params -> {
        ItemStack item = new ItemStack(Material.DIAMOND_SWORD);
        Utility.setName(item, ChatColor.YELLOW + "" + ChatColor.BOLD + "Heaven's Blade");
        Utility.addLoreLine(item, "A godly blade that weilds incredible desctructive power");
        item.addEnchantment(Enchantment.DAMAGE_ALL, 5);
        item.addEnchantment(Enchantment.KNOCKBACK, 2);
        item.addEnchantment(Enchantment.FIRE_ASPECT, 2);
        item.addEnchantment(Enchantment.DURABILITY, 3);
        item.addEnchantment(Enchantment.MENDING, 1);
        params.rewardee.sendMessage(ChatColor.YELLOW + "" + ChatColor.BOLD + "You got Heaven's Blade!");
        return Collections.singletonList(item);
    }, params -> Utility.setName(Material.DIAMOND_SWORD, ChatColor.YELLOW + "" + ChatColor.BOLD + "Heaven's Blade")),

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
    }, params -> Utility.setName(Material.IRON_CHESTPLATE, ChatColor.AQUA + "Full Iron Combat Gear")),

    IRON_TOOLSET (params -> {
        ArrayList<ItemStack> rewards = new ArrayList<>();
        rewards.add(new ItemStack(Material.IRON_PICKAXE));
        rewards.add(new ItemStack(Material.IRON_AXE));
        rewards.add(new ItemStack(Material.IRON_SWORD));
        rewards.add(new ItemStack(Material.IRON_SPADE));
        rewards.add(new ItemStack(Material.IRON_HOE));
        params.rewardee.sendMessage("You got a full iron toolset!");
        return rewards;
    }, params -> Utility.setName(Material.IRON_PICKAXE, ChatColor.AQUA + "Full Iron Toolset")),

    DIAMONDS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " Diamonds!");
        else
            params.rewardee.sendMessage("You got a Diamond!");
        return Collections.singletonList(new ItemStack(Material.DIAMOND, params.amountToGive));
    }, params -> {
        if (params.amountToGive > 1)
            return Utility.setName(Material.DIAMOND, ChatColor.DARK_AQUA + "" + params.amountToGive + " Diamonds");
        else
            return Utility.setName(Material.DIAMOND, ChatColor.DARK_AQUA + "1 Diamond");
    }),

    EMERALDS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage(ChatColor.GREEN + "You got " + params.amountToGive + " Emeralds!");
        else
            params.rewardee.sendMessage(ChatColor.GREEN + "You got an Emerald!");
        return Collections.singletonList(new ItemStack(Material.EMERALD, params.amountToGive));
    }, params -> {
        if (params.amountToGive > 1)
            return Utility.setName(Material.EMERALD, ChatColor.GREEN + "" + params.amountToGive + " Emeralds");
        else
            return Utility.setName(Material.EMERALD, ChatColor.GREEN + "1 Emerald");
    }),

    IRON_BARS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " iron ingots!");
        else
            params.rewardee.sendMessage("You got an iron ingot!");
        return Collections.singletonList(new ItemStack(Material.IRON_INGOT, params.amountToGive));
    }, params -> {
        if (params.amountToGive > 1)
            return Utility.setName(Material.IRON_INGOT, ChatColor.DARK_AQUA + "" + params.amountToGive + " Iron Ingots");
        else
            return Utility.setName(Material.IRON_INGOT, ChatColor.DARK_AQUA + "1 Iron Ingot");
    }),

    GOLD_BARS (params -> {
        if (params.amountToGive > 1)
            params.rewardee.sendMessage("You got " + params.amountToGive + " gold ingots!");
        else
            params.rewardee.sendMessage("You got a gold ingot!");
        return Collections.singletonList(new ItemStack(Material.GOLD_INGOT, params.amountToGive));
    }, params -> {
        if (params.amountToGive > 1)
            return Utility.setName(Material.IRON_INGOT, ChatColor.DARK_AQUA + "" + params.amountToGive + " Gold Ingots");
        else
            return Utility.setName(Material.IRON_INGOT, ChatColor.DARK_AQUA + "1 Gold Ingot");
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
    }, params -> Utility.setName(Material.REDSTONE_ORE, ChatColor.AQUA + "" + params.amountToGive + "xAssorted Raw Ore Blocks")),

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
    }, params -> Utility.setName(Material.COAL, ChatColor.AQUA + "" + params.amountToGive + "xAssorted Ores")),

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
    }, params -> Utility.setName(Material.CHEST, ChatColor.GREEN + "" + params.amountToGive + "xAssorted Items")),

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
        multipliers.put(Material.POTATO_ITEM, 7);
        multipliers.put(Material.CARROT_ITEM, 4);
        for (Material key : multipliers.keySet())
        {
            int basecount = params.amountToGive * multipliers.get(key);
            ItemStack item = new ItemStack(key, Utility.randomInt(basecount, basecount*2));
            rewards.add(item);
        }
        params.rewardee.sendMessage(ChatColor.GREEN + "You got an assorted planter set.");
        return rewards;
    }, params -> Utility.setName(Material.SAPLING, ChatColor.DARK_GREEN + "" + params.amountToGive + "xAssorted Plants")),

    MONEY (params -> {
        Utility.modifyBalance(params.rewardee, new BigDecimal(params.amountToGive));
        params.rewardee.sendMessage("You got $" + params.amountToGive + "!");
        return Collections.singletonList(Utility.setName(Material.PAPER, ChatColor.DARK_AQUA + "$" + params.amountToGive + " bank invoice"));
    }, params -> Utility.setName(Material.GOLD_INGOT, ChatColor.DARK_AQUA + "$" + params.amountToGive)),

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
        if (params.amountToGive > 1)
            return Utility.setName(Material.GOLD_NUGGET, ChatColor.DARK_AQUA + "" + params.amountToGive + " Individually Placed Gold Nuggets");
        else
            return Utility.setName(Material.GOLD_NUGGET, ChatColor.DARK_AQUA + "1 Gold Nugget");
    }),

    // bad prizes

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
                    .with(Utility.randomElement(effectTypes))
                    .withColor(Color.fromRGB(Utility.randomInt(128,255), Utility.randomInt(128,255), Utility.randomInt(128,255)))
                    .build();
            meta.addEffect(effect);
            firework.setFireworkMeta(meta);
        }
        params.rewardee.sendMessage(ChatColor.GOLD + "You got a fireworks show! Yaayyy!!");
        return null;
    }, params -> Utility.setName(Material.FIREWORK, ChatColor.GOLD + "A Fireworks Show")),

    BOOM_TIME (params -> {
        Chest chest = (Chest) params.chestBlock.getState();
        for (int i=0;i<chest.getInventory().getSize();i++)
            chest.getInventory().setItem(i, Utility.setName(Material.TNT, ChatColor.RED + "BOOM TIME!"));
        params.rewardee.sendMessage(ChatColor.RED + "Boom Time!");
        params.plugin.activeJobs.add(new BoomTimeJob(params.chestBlock, 8, 6, params.amountToGive));
        return null;
    }, params -> Utility.setName(Material.TNT, ChatColor.RED + "Boom Time!")),

    PANDORAS_BOX (params -> {
        Location spawnPos = params.chestBlock.getLocation().add(0, 1, 0);
        for (int i=0;i<params.amountToGive*2;i++)
            spawnPos.getWorld().spawnEntity(spawnPos.add(Utility.randomInsideUnitCircle()), EntityType.ZOMBIE);
        for (int i=0;i<params.amountToGive;i++)
            spawnPos.getWorld().spawnEntity(spawnPos.add(Utility.randomInsideUnitCircle()), EntityType.SKELETON);
        for (int i=0;i<params.amountToGive;i++)
            spawnPos.getWorld().spawnEntity(spawnPos.add(Utility.randomInsideUnitCircle()), EntityType.SPIDER);
        for (int i=0;i<params.amountToGive;i++)
            spawnPos.getWorld().spawnEntity(spawnPos.add(Utility.randomInsideUnitCircle()), EntityType.CAVE_SPIDER);
        for (int i=0;i<params.amountToGive*4;i++)
            spawnPos.getWorld().spawnEntity(spawnPos.add(Utility.randomInsideUnitCircle()), EntityType.BAT);
        spawnPos.getWorld().spawnEntity(spawnPos.add(0, 5, 0), EntityType.LIGHTNING);
        params.rewardee.sendMessage(ChatColor.DARK_PURPLE + "You opened pandora's box!");
        return null;
    }, params -> Utility.setName(Material.ENDER_CHEST, ChatColor.DARK_PURPLE + "Pandora's Box")),

    NOTHING (params -> {
        params.rewardee.sendMessage(ChatColor.DARK_GRAY + "You got nothing.");
        return null;
    }, params -> Utility.setName(Material.THIN_GLASS, ChatColor.DARK_GRAY + "Nothing"));

    private PrizeActor action;
    private PrizeVisualizer visualisation;
    Prize(PrizeActor action, PrizeVisualizer visualisation)
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
        for (ItemStack i : rewardItems)
        {
            if (i.getAmount() == 1 && Utility.getLore(i).size() > 0)
                Utility.addLoreLine(i, getLoreTag());
        }
        Utility.arrangeItemsInExistingInventory(chest.getInventory(), rewardItems);
    }

    public String toString()
    {
        return name().toLowerCase();
    }

    public ItemStack getVisualisation(LootCrate plugin, Player rewardee, int amount, Block chestBlock)
    {
        return visualisation.getVisualisation(new RewardActionParameter(plugin, rewardee, amount, chestBlock));
    }

    public String getLoreTag()
    {
        return ChatColor.BLACK + toString().toLowerCase();
    }

    public static boolean itemIsPrize(ItemStack item, Prize prize)
    {
        return Utility.itemHasLoreLine(item, prize.getLoreTag());
        //TODO There was another check here that I wanted to add... but i've forgotten what it was
        // maybe i'll remember it someday
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

    interface PrizeVisualizer
    {
        ItemStack getVisualisation(RewardActionParameter parameters);
    }

    interface PrizeActor
    {
        List<ItemStack> enactReward(RewardActionParameter parameters);
    }
}
