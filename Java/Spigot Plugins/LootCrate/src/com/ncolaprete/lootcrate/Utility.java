package com.ncolaprete.lootcrate;

import com.earth2me.essentials.api.NoLoanPermittedException;
import com.earth2me.essentials.api.UserDoesNotExistException;
import net.ess3.api.Economy;
import net.minecraft.server.v1_9_R1.TileEntityChest;
import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.craftbukkit.v1_9_R1.block.CraftChest;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.util.BlockIterator;
import org.bukkit.util.Vector;

import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

class Utility
{
    public static final char SerializationDelimiter = '?';

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

    public static List<String> getLore(ItemStack item)
    {
        ItemMeta meta = item.getItemMeta();
        if (meta == null)
            return new ArrayList<>();
        List<String> lore = meta.getLore();
        if (lore == null)
            return new ArrayList<>();
        return lore;
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

    public static Vector randomInsideUnitCircle()
    {
        double x, y;
        do {
            x = Math.random() * 2 - 1;
            y = Math.random() * 2 - 1;
        } while (x*x + y*y > 1);
        return new Vector(x, 0, y);
    }

    public static <T> T randomElement(Collection<T> c)
    {
        int index = randomInt(0, c.size());
        for (T elem : c)
        {
            if (index == 0)
                return elem;
            index--;
        }
        return c.iterator().next();
    }

    public static <T> T randomElement(T[] a)
    {
        return a[randomInt(0, a.length)];
    }

    public static int randomWeightedIndex(List<Double> weights)
    {
        double sum = 0;
        for (double f : weights)
            sum += f;
        if (sum == 0)
            return -1;
        double rand = Math.random() * sum;
        for (int i=0;i<weights.size();i++)
        {
            if (rand < weights.get(i))
                return i;
            rand -= weights.get(i);
        }
        return weights.size() - 1;
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

    public static String serializeLocation(Location loc)
    {
        StringBuilder sb = new StringBuilder();
        sb.append(loc.getWorld().getName());
        sb.append(SerializationDelimiter);
        sb.append(loc.getBlockX());
        sb.append(SerializationDelimiter);
        sb.append(loc.getBlockY());
        sb.append(SerializationDelimiter);
        sb.append(loc.getBlockZ());
        return sb.toString();
    }

    public static Location deserializeLocation(Server server, String serial)
    {
        String[] parts = serial.split(Pattern.quote(SerializationDelimiter + ""));
        World world = server.getWorld(parts[0]);
        int x = Integer.parseInt(parts[1]);
        int y = Integer.parseInt(parts[2]);
        int z = Integer.parseInt(parts[3]);
        return world.getBlockAt(x, y, z).getLocation();
    }

    public static String formatVector(Vector v)
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

    public static Location getDefaultSpawn(JavaPlugin plugin)
    {
        return plugin.getServer().getWorlds().get(0).getSpawnLocation();
    }

    public static Block getHighestSolidBlock(World world, int x, int z)
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

    public static List<ItemStack> separateItemStacks(List<ItemStack> items)
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

    public static List<Block> getSurroundingBlocks(Block block, boolean sides, boolean diagonals, boolean corners)
    {
        ArrayList<Block> surrounds = new ArrayList<>();
        EnumSet<BlockFace> cardinalFaces = EnumSet.of(BlockFace.NORTH, BlockFace.SOUTH, BlockFace.EAST, BlockFace.WEST, BlockFace.UP, BlockFace.DOWN);
        if (sides)
        {
            surrounds.addAll(cardinalFaces.stream().map(block::getRelative).collect(Collectors.toList()));
            surrounds.add(block.getRelative(BlockFace.UP));
            surrounds.add(block.getRelative(BlockFace.DOWN));
        }
        if (diagonals)
        {
            EnumSet<BlockFace> diagonalFaces = EnumSet.of(BlockFace.NORTH_EAST, BlockFace.NORTH_WEST, BlockFace.SOUTH_EAST, BlockFace.SOUTH_WEST);
            for (BlockFace face : cardinalFaces)
            {
                surrounds.add(block.getRelative(face).getRelative(BlockFace.UP));
                surrounds.add(block.getRelative(face).getRelative(BlockFace.DOWN));
            }
            surrounds.addAll(diagonalFaces.stream().map(block::getRelative).collect(Collectors.toList()));
        }
        if (corners)
        {
            for (BlockFace fa : EnumSet.of(BlockFace.NORTH, BlockFace.SOUTH))
            {
                for (BlockFace fb : EnumSet.of(BlockFace.EAST, BlockFace.WEST))
                {
                    surrounds.addAll(EnumSet.of(BlockFace.UP, BlockFace.DOWN).stream().map(fc -> block.getRelative(fa).getRelative(fb).getRelative(fc)).collect(Collectors.toList()));
                }
            }
        }
        return surrounds;
    }

    public static boolean isCorrectTool(Material tool, Material block)
    {
        EnumSet<Material> Pickaxes = EnumSet.of(Material.WOOD_PICKAXE, Material.STONE_PICKAXE, Material.IRON_PICKAXE, Material.DIAMOND_PICKAXE);
        EnumSet<Material> Axes = EnumSet.of(Material.WOOD_AXE, Material.STONE_AXE, Material.IRON_AXE, Material.DIAMOND_AXE);
        EnumSet<Material> Shovels = EnumSet.of(Material.WOOD_SPADE, Material.STONE_SPADE, Material.IRON_SPADE, Material.DIAMOND_SPADE);
        EnumSet<Material> validAxeBlocks = EnumSet.of(
                Material.WOOD_DOOR, Material.ACACIA_DOOR, Material.BIRCH_DOOR,
                Material.DARK_OAK_DOOR, Material.JUNGLE_DOOR, Material.SPRUCE_DOOR,
                Material.TRAP_DOOR, Material.CHEST, Material.WORKBENCH,
                Material.FENCE, Material.FENCE_GATE, Material.JUKEBOX,
                Material.WOOD, Material.LOG, Material.LOG_2, Material.BOOKSHELF,
                Material.JACK_O_LANTERN, Material.PUMPKIN, Material.SIGN_POST,
                Material.WALL_SIGN, Material.NOTE_BLOCK, Material.WOOD_PLATE,
                Material.DAYLIGHT_DETECTOR, Material.DAYLIGHT_DETECTOR_INVERTED,
                Material.HUGE_MUSHROOM_1, Material.HUGE_MUSHROOM_2, Material.VINE);
        EnumSet<Material> validShovelBlocks = EnumSet.of(
                Material.CLAY, Material.SOIL, Material.GRASS, Material.GRASS_PATH,
                Material.GRAVEL, Material.MYCEL, Material.DIRT, Material.SAND,
                Material.SOUL_SAND, Material.SNOW_BLOCK);
        EnumSet<Material> noValidToolBlocks = EnumSet.of(
                Material.LEAVES, Material.LEAVES_2,
                Material.MELON_STEM, Material.PUMPKIN_STEM,
                Material.CARROT, Material.POTATO, Material.SUGAR_CANE_BLOCK,
                Material.NETHER_STALK, Material.LONG_GRASS, Material.WATER_LILY,
                Material.CHORUS_FLOWER, Material.YELLOW_FLOWER, Material.GLASS,
                Material.THIN_GLASS, Material.BEDROCK);
        if (tool == Material.SHEARS && EnumSet.of(Material.LEAVES, Material.LEAVES_2).contains(block))
            return true;
        if (Pickaxes.contains(tool))
        {
            return !validAxeBlocks.contains(block) && !validShovelBlocks.contains(block) && !noValidToolBlocks.contains(block);
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

    public static boolean blockInBoundsInclusive(Block minBounds, Block maxBounds, Block check)
    {
        if (!(check.getX() >= minBounds.getX() && check.getX() <= maxBounds.getX()))
            return false;
        if (!(check.getY() >= minBounds.getY() && check.getY() <= maxBounds.getY()))
            return false;
        if (!(check.getZ() >= minBounds.getZ() && check.getZ() <= maxBounds.getZ()))
            return false;
        return true;
    }

    public static BlockFace getBlockFaceIsLookingAt(Player ply)
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

    public static boolean isOnGround(Player ply)
    {
        return ((Entity) ply).isOnGround();
    }

    public static boolean reduceDurability(Player ply, ItemStack tool, int amount)
    {
        tool.setDurability((short) (tool.getDurability() + amount));
        if (tool.getDurability() >= tool.getType().getMaxDurability())
        {
            ply.playSound(ply.getLocation(), Sound.ENTITY_ITEM_BREAK, 1, 1);
            ply.spawnParticle(Particle.ITEM_CRACK, ply.getLocation(), 20);
            ply.getInventory().remove(tool);
            return true;
        }
        return false;
    }

    public static BigDecimal getBalance(Player ply)
    {
        try {
            return Economy.getMoneyExact(ply.getName());
        } catch (UserDoesNotExistException e) {
            return BigDecimal.ZERO;
        }
    }

    public static boolean modifyBalance(Player ply, BigDecimal amount)
    {
        try {
            int comparison = amount.compareTo(BigDecimal.ZERO);
            if (comparison == 1)
                Economy.add(ply.getName(), amount);
            else if (comparison == -1)
                Economy.substract(ply.getName(), amount.abs());
        } catch (UserDoesNotExistException e) {
            return false;
        } catch (NoLoanPermittedException e) {
            return false;
        }
        return true;
    }

    public static Block getTargetBlock(Player ply)
    {
        BlockIterator iter = new BlockIterator(ply);
        do {
            Block b = iter.next();
            if (b.getType() != Material.AIR)
                return b;
        } while (iter.hasNext());
        return null;
    }

    public static Inventory arrangeItems(String invName, List<ItemStack> items)
    {
        int invSize = items.size() - (items.size()%9) + 9;

        Inventory inv = Bukkit.createInventory(null, invSize, invName);
        return arrangeItemsInExistingInventory(inv, items);
    }

    public static Inventory arrangeItemsWithBorder(Inventory inv, List<ItemStack> items, ItemStack border)
    {
        return arrangeItemsInExistingInventory(inv, items); // Placeholder for now
    }

    public static Inventory arrangeItemsInExistingInventory(Inventory inv, List<ItemStack> items)
    {
        return arrangeItemsInExistingInventory(inv, items, 9);
    }

    public static Inventory arrangeItemsInExistingInventory(Inventory inv, List<ItemStack> items, int invWidth)
    {
        int centerSlot = (int)((invWidth-1) / 2.0);
        int rowCount = inv.getSize()/invWidth;
        int centerRow = rowCount/2;
        double rowOffset = 0;
        int rowOffsetDirection = -1;
        ListIterator<ItemStack> itemsListIter = items.listIterator();
        if (rowCount % 2 == 0)
        {
            for (int r = 0; r < rowCount/2; r++)
            {
                rowOffset = r;
                double slotOffset = 0;
                int slotOffsetDirection = 1;
                for (int c = 0; c < invWidth; c++)
                {
                    slotOffset += 0.5;
                    slotOffsetDirection *= -1;
                    int totalSlotOffset = (int)slotOffset * slotOffsetDirection;
                    for (int j=0;j<2;j++)
                    {
                        if (!itemsListIter.hasNext())
                            return inv;
                        int totalRowOffset = (int)rowOffset * rowOffsetDirection;
                        int slot = (centerRow - totalRowOffset)*invWidth + centerSlot + totalSlotOffset;
                        inv.setItem(slot, itemsListIter.next());
                        rowOffsetDirection *= -1;
                        rowOffset += rowOffsetDirection;
                    }
                }
            }
        }
        else
        {
            // fill out the middle row
            double slotOffset = 0;
            int slotOffsetDirection = 1;
            for (int c = 0; c < invWidth; c++)
            {
                if (!itemsListIter.hasNext())
                    return inv;
                slotOffset += 0.5;
                slotOffsetDirection *= -1;
                int totalSlotOffset = (int)slotOffset * slotOffsetDirection;
                int slot = centerRow * invWidth + centerSlot + totalSlotOffset;
                inv.setItem(slot, itemsListIter.next());
            }

            // fill out subsequent rows
            for (int r = 1; r <= (rowCount-1)/2; r++)
            {
                rowOffset = r;
                slotOffset = 0;
                slotOffsetDirection = 1;
                for (int c = 0; c < invWidth; c++)
                {
                    slotOffset += 0.5;
                    slotOffsetDirection *= -1;
                    int totalSlotOffset = (int)slotOffset * slotOffsetDirection;
                    for (int j=0;j<2;j++)
                    {
                        if (!itemsListIter.hasNext())
                            return inv;
                        int slot = (int)(centerRow + rowOffset)*invWidth + centerSlot + totalSlotOffset;
                        inv.setItem(slot, itemsListIter.next());
                        rowOffset *= -1;
                    }
                }
            }
        }
        return inv;
    }
}