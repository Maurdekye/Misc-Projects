package mainpack;

import org.bukkit.block.Block;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.block.SignChangeEvent;
import org.bukkit.event.inventory.InventoryCloseEvent;
import org.bukkit.inventory.InventoryHolder;
import org.bukkit.scheduler.BukkitRunnable;

import java.util.HashMap;

public class Listen implements Listener {

    Rewind parent;
    Listen(Rewind parent) {
        this.parent = parent;
    }

    @EventHandler
    public void Break(final BlockBreakEvent event) {
        HashMap<Integer, BlockInformation> userPreSteps = new HashMap<>();
        HashMap<Box, BlockInformation> areaPreSteps = new HashMap<>();
        for (int i=0;i<parent.userRecordings.size();i++) {
            Recording rec = parent.userRecordings.get(0);
            if (rec.owner.equals(event.getPlayer().getUniqueId())) {
                userPreSteps.put(i, new BlockInformation(event.getBlock()));
            }
        }
        for (Box b : parent.areaRecordings.keySet()) {
            if (b.contains(event.getBlock().getLocation().toVector())) {
                areaPreSteps.put(b, new BlockInformation(event.getBlock()));
            }
        }

        final HashMap<Integer, BlockInformation> fUserPreSteps = userPreSteps;
        final HashMap<Box, BlockInformation> fAreaPreSteps = areaPreSteps;

        new BukkitRunnable() {
            public void run() {
                for (int i : fUserPreSteps.keySet()) {
                    parent.userRecordings.get(i).addStep(
                            fUserPreSteps.get(i),
                            new BlockInformation(event.getBlock()),
                            event.getBlock().getLocation(),
                            parent.ticker);
                }
                for (Box b : fAreaPreSteps.keySet()) {
                    parent.areaRecordings.get(b).addStep(
                            fAreaPreSteps.get(b),
                            new BlockInformation(event.getBlock()),
                            event.getBlock().getLocation(),
                            parent.ticker);
                }
            }
        }.runTaskLater(parent, 1);
    }

    @EventHandler
    public void Break(final BlockPlaceEvent event) {
        HashMap<Integer, BlockInformation> userPreSteps = new HashMap<>();
        HashMap<Box, BlockInformation> areaPreSteps = new HashMap<>();
        for (int i=0;i<parent.userRecordings.size();i++) {
            Recording rec = parent.userRecordings.get(0);
            if (rec.owner.equals(event.getPlayer().getUniqueId())) {
                userPreSteps.put(i, new BlockInformation(event.getBlock()));
            }
        }
        for (Box b : parent.areaRecordings.keySet()) {
            if (b.contains(event.getBlock().getLocation().toVector())) {
                areaPreSteps.put(b, new BlockInformation(event.getBlock()));
            }
        }

        final HashMap<Integer, BlockInformation> fUserPreSteps = userPreSteps;
        final HashMap<Box, BlockInformation> fAreaPreSteps = areaPreSteps;

        new BukkitRunnable() {
            public void run() {
                for (int i : fUserPreSteps.keySet()) {
                    parent.userRecordings.get(i).addStep(
                            fUserPreSteps.get(i),
                            new BlockInformation(event.getBlock()),
                            event.getBlock().getLocation(),
                            parent.ticker);
                }
                for (Box b : fAreaPreSteps.keySet()) {
                    parent.areaRecordings.get(b).addStep(
                            fAreaPreSteps.get(b),
                            new BlockInformation(event.getBlock()),
                            event.getBlock().getLocation(),
                            parent.ticker);
                }
            }
        }.runTaskLater(parent, 1);
    }

    @EventHandler
    public void Break(final InventoryCloseEvent event) {
        InventoryHolder holder = event.getInventory().getHolder();
        if (!(holder instanceof Block)) return;
        final Block block = (Block) holder;
        HashMap<Integer, BlockInformation> userPreSteps = new HashMap<>();
        HashMap<Box, BlockInformation> areaPreSteps = new HashMap<>();
        for (int i=0;i<parent.userRecordings.size();i++) {
            Recording rec = parent.userRecordings.get(0);
            if (rec.owner.equals(event.getPlayer().getUniqueId())) {
                userPreSteps.put(i, new BlockInformation(block));
            }
        }
        for (Box b : parent.areaRecordings.keySet()) {
            if (b.contains(block.getLocation().toVector())) {
                areaPreSteps.put(b, new BlockInformation(block));
            }
        }

        final HashMap<Integer, BlockInformation> fUserPreSteps = userPreSteps;
        final HashMap<Box, BlockInformation> fAreaPreSteps = areaPreSteps;

        new BukkitRunnable() {
            public void run() {
                for (int i : fUserPreSteps.keySet()) {
                    parent.userRecordings.get(i).addStep(
                            fUserPreSteps.get(i),
                            new BlockInformation(block),
                            block.getLocation(),
                            parent.ticker);
                }

                for (Box b : fAreaPreSteps.keySet()) {
                    parent.areaRecordings.get(b).addStep(
                            fAreaPreSteps.get(b),
                            new BlockInformation(block),
                            block.getLocation(),
                            parent.ticker);
                }
            }
        }.runTaskLater(parent, 1);
    }

    @EventHandler
    public void Break(final SignChangeEvent event) {
        HashMap<Integer, BlockInformation> userPreSteps = new HashMap<>();
        HashMap<Box, BlockInformation> areaPreSteps = new HashMap<>();
        for (int i=0;i<parent.userRecordings.size();i++) {
            Recording rec = parent.userRecordings.get(0);
            if (rec.owner.equals(event.getPlayer().getUniqueId())) {
                userPreSteps.put(i, new BlockInformation(event.getBlock()));
            }
        }
        for (Box b : parent.areaRecordings.keySet()) {
            if (b.contains(event.getBlock().getLocation().toVector())) {
                areaPreSteps.put(b, new BlockInformation(event.getBlock()));
            }
        }

        final HashMap<Integer, BlockInformation> fUserPreSteps = userPreSteps;
        final HashMap<Box, BlockInformation> fAreaPreSteps = areaPreSteps;

        new BukkitRunnable() {
            public void run() {
                for (int i : fUserPreSteps.keySet()) {
                    parent.userRecordings.get(i).addStep(
                            fUserPreSteps.get(i),
                            new BlockInformation(event.getBlock()),
                            event.getBlock().getLocation(),
                            parent.ticker);
                }

                for (Box b : fAreaPreSteps.keySet()) {
                    parent.areaRecordings.get(b).addStep(
                            fAreaPreSteps.get(b),
                            new BlockInformation(event.getBlock()),
                            event.getBlock().getLocation(),
                            parent.ticker);
                }
            }
        }.runTaskLater(parent, 1);
    }
}
