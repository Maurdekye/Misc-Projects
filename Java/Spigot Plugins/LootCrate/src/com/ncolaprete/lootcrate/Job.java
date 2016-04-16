package com.ncolaprete.lootcrate;

import org.bukkit.Effect;
import org.bukkit.GameMode;
import org.bukkit.Material;
import org.bukkit.Sound;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.material.MaterialData;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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
    Material initialReplacementMaterial;
    byte initialData;
    byte initialReplacementData;
    Player ply;
    ArrayList<Block> currentSet;
    int progressThroughSet;
    int blocksModified;
    int modificationsPerAction;
    int maxModification;
    boolean finished = false;

    public TransmogrificationJob(Block initialBlock, Player ply, int modificationsPerAction, int maxModification)
    {
        this.initialBlock = initialBlock;
        this.initialMaterial = initialBlock.getType();
        this.initialReplacementMaterial = ply.getInventory().getItemInOffHand().getType();
        this.initialData = initialBlock.getData();
        this.initialReplacementData = (byte)ply.getInventory().getItemInOffHand().getDurability();
        if (initialMaterial == Material.LEAVES || initialMaterial == Material.LEAVES_2)
            initialData = (byte) (initialData % 4);
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
            if (isDone())
            {
                return;
            }
            ItemStack offhanditem = ply.getInventory().getItemInOffHand();
            if (offhanditem == null)
                return;
            if (progressThroughSet >= currentSet.size()) {
                progressThroughSet = 0;
                getNewSet();
            } else if (currentSet.get(progressThroughSet).getType() != Material.BEDROCK) {
                if (ply.getGameMode() != GameMode.CREATIVE)
                {
                    if (offhanditem.getAmount() == 1)
                    {
                        ply.getInventory().setItem(40, null);
                        finished = true;
                    }
                    else
                        offhanditem.setAmount(offhanditem.getAmount() - 1);
                }
                if (offhanditem.getType() != initialReplacementMaterial)
                {
                    finished = true;
                    break;
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
                if (initialMaterial != nblock.getType())
                    continue;
                if (initialMaterial == Material.LEAVES || initialMaterial == Material.LEAVES_2)
                {
                    if (initialData != nblock.getData() % 4)
                        continue;
                }
                else
                {
                    if (initialData != nblock.getData())
                        continue;
                }
                if (nblock.getType() == Material.BEDROCK)
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
                finished;
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

class BoomTimeJob implements Job
{
    private Block chest;
    private int clicks;
    private int clickdelay;
    private int explosivePower;

    private int counter;

    public BoomTimeJob(Block chest, int clicks, int clickdelay, int explosivePower)
    {
        this.chest = chest;
        this.clicks = clicks;
        this.clickdelay = clickdelay;
        this.explosivePower = explosivePower;
    }

    public void update()
    {
        counter++;
        if (chest.getType() != Material.CHEST)
            counter = clickdelay * clicks + 1;
        if (counter % clickdelay == 0)
        {
            chest.getWorld().playSound(chest.getLocation(), Sound.BLOCK_LEVER_CLICK, 10, 1);
        }
        if (counter == clicks * clickdelay)
        {
            chest.setType(Material.AIR);
            chest.getWorld().createExplosion(chest.getLocation(), explosivePower);
        }
    }

    public boolean isDone()
    {
        return counter > clicks * clickdelay;
    }
}

class DematerializationJob implements Job
{
    public List<Block> currentSet;
    int progressThroughSet;

    public DematerializationJob(List<Block> currentActiveSet)
    {
        this.currentSet = currentActiveSet;
    }

    public void update()
    {
        do {
            if (progressThroughSet >= currentSet.size())
            {
                progressThroughSet = 0;
                getNewSet();
                return;
            }
            if (currentSet.get(progressThroughSet).getType() == Material.AIR ||
                    currentSet.get(progressThroughSet).getType() == Material.BEDROCK)
            {
                progressThroughSet++;
                continue;
            }
            if (currentSet.get(progressThroughSet).getType() != Material.BEDROCK) {
                currentSet.get(progressThroughSet).setType(Material.AIR);
                progressThroughSet++;
            }
        } while (false);
    }

    public void getNewSet()
    {
        ArrayList<Block> nextUpdate = new ArrayList<>();
        for (Block currentblock : currentSet)
        {
            for (Block nblock : Utility.getSurroundingBlocks(currentblock, true, false, false))
            {
                if (nblock.getType() == Material.AIR)
                    continue;
                if (nblock.getType() == Material.BEDROCK)
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
        return currentSet.size() == 0;
    }
}