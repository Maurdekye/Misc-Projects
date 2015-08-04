package mainpack;

import org.bukkit.*;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.block.BlockState;
import org.bukkit.block.Chest;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.inventory.CraftItemEvent;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.inventory.InventoryEvent;
import org.bukkit.event.player.*;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.ShapedRecipe;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.material.MaterialData;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.potion.PotionEffect;
import org.bukkit.potion.PotionEffectType;
import org.bukkit.scheduler.BukkitRunnable;
import org.bukkit.util.BlockIterator;
import org.bukkit.util.Vector;

import java.util.*;

public class PluginTestingV2 extends JavaPlugin implements Listener {

    IngameConfig icfg;
    ShapedRecipe testrecipe;

    public String BOOT_TAG = ChatColor.GRAY + "" + ChatColor.ITALIC + "Super Speed!";

    public void onEnable() {

        testrecipe = new ShapedRecipe(new ItemStack(Material.STONE, 8));
        testrecipe.shape("***", "*U*", "***");
        testrecipe.setIngredient('U', Material.LAVA_BUCKET);
        testrecipe.setIngredient('*', Material.COBBLESTONE);
        Bukkit.addRecipe(testrecipe);

        try {
            icfg = new IngameConfig(this, Arrays.asList("somekey", "someotherkey", "athirdkey"));
        } catch (Exception e) {
            e.printStackTrace();
        }
        Bukkit.getPluginManager().registerEvents(this, this);
        Bukkit.getScheduler().scheduleSyncRepeatingTask(this, new Runnable() {
            public void run() {
                for (Player ply : Bukkit.getOnlinePlayers()) {
                    if (ply.isBlocking() && ply.getItemInHand().getType() == Material.DIAMOND_SWORD) {
                        BlockIterator iter = new BlockIterator(ply, 256);
                        Block look = iter.next();
                        while (look.getType() == Material.AIR && iter.hasNext())
                            look = iter.next();
                        for (int i=0;i<5;i++) ply.getWorld().spawnArrow(look.getLocation().add(new Vector(0.5, 40, 0.5)), new Vector(0, -2, 0), 2 , 3);
                    }
                }
            }
        }, 0, 1);
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        Player ply = sender instanceof Player ? (Player) sender : null;
        if (ply == null) {
            sender.sendMessage("nope.");
            return true;
        }
        if (command.getName().equals("whatvalue")) {
            int a = icfg.getValue("somekey");
            int b = icfg.getValue("someotherkey");
            sender.sendMessage(a + " + " + b + " = " + (a + b) + ", and " + icfg.getValue("athirdkey"));
        }

        else if (command.getName().equals("showmenu")) {
            icfg.showMenu(ply);
        }

        else if (command.getName().equals("giveboots")) {
            ItemStack boots = new ItemStack(Material.DIAMOND_BOOTS);
            ItemMeta bootMeta = boots.getItemMeta();
            bootMeta.setDisplayName("Boots of Walking Really, Really Fast");
            bootMeta.setLore(Arrays.asList(BOOT_TAG));
            boots.setItemMeta(bootMeta);
            ply.getInventory().addItem(boots);
        }

        return true;
    }

    @EventHandler
    public void Break(BlockBreakEvent event) {
        event.setCancelled(true);
        Material type = event.getBlock().getType();
        event.getBlock().setType(Material.CHEST);
        ((Chest) event.getBlock().getState()).getInventory().setItem(14, new ItemStack(type));
    }

    @EventHandler
    public void Interact(PlayerInteractEvent event) {
        if (event.getAction() == Action.LEFT_CLICK_AIR || event.getAction() == Action.LEFT_CLICK_BLOCK) {
            Player ply = event.getPlayer();
            if (event.getItem() == null) {

            } else if (event.getItem().getType() == Material.STICK) {
                Location dir = ply.getLocation();
                event.getPlayer().sendMessage(dir.getYaw() + "");
            } else if (event.getItem().getType() == Material.BLAZE_ROD) {
                ply.getEyeLocation().getBlock().getRelative(getVirtualDirection(ply.getLocation())).setType(Material.REDSTONE_BLOCK);
                double yaw = ply.getLocation().getYaw();
                ply.sendMessage((yaw > 180 ?  yaw - 360 : yaw) + "");
            } else if (event.getItem().getType() == Material.RED_ROSE) {
                for (Block b : getLastTwoTargets(ply, 48))
                    b.setType(Material.REDSTONE_BLOCK);
            }
        } else if (event.getAction() == Action.RIGHT_CLICK_AIR || event.getAction() == Action.RIGHT_CLICK_BLOCK) {
            Player ply = event.getPlayer();
            if (event.getItem() == null) {

            } else if (event.getItem().getType() == Material.STICK) {
                Location point = getTargetPoint(ply, 64);
                if (point == null) return;
                for (int i=0;i<20;i++)
                    ParticleEffects.sendToLocation(ParticleEffects.CRITICAL_HIT, point, 0.1F, 0.1F, 0.1F, 0.1F, 1);
            }
        }
    }

    @EventHandler
    public void Craft(final CraftItemEvent event) {
       if (event.getRecipe().equals(testrecipe)) {
           event.getInventory().addItem(new ItemStack(Material.BUCKET));
       }
    }

    @EventHandler
    public void Login(PlayerLoginEvent event) {
        Bukkit.getLogger().info("Player Logging In: " + event.getPlayer().getDisplayName());
    }
    @EventHandler
    public void Join(PlayerJoinEvent event) {
        Bukkit.getLogger().info("Player Joined: " + event.getPlayer().getDisplayName());
    }
    @EventHandler
    public void Leave(PlayerQuitEvent event) {
        Bukkit.getLogger().info("Player Left: " + event.getPlayer().getDisplayName());
    }

    @EventHandler
    public void Inv(final InventoryClickEvent event) {
        if (!(event.getWhoClicked() instanceof Player)) return;
        new BukkitRunnable() {
            public void run() {
                Player user = (Player) event.getWhoClicked();
                ItemStack boots = user.getInventory().getBoots();
                try {
                    if (!boots.getItemMeta().getLore().get(0).equals(BOOT_TAG)) {
                        throw new NullPointerException();
                    } else {
                        user.addPotionEffect(new PotionEffect(PotionEffectType.SPEED, 999999, 4));
                    }
                } catch (NullPointerException e) {
                    user.removePotionEffect(PotionEffectType.SPEED);
                }
            }
        }.runTaskLater(this, 1);
    }

    public double roundDetail(double toRound, int decimals) {
        toRound *= 10^decimals;
        toRound = Math.round(toRound);
        toRound /= 10^decimals;
        return toRound;
    }

    public BlockFace getVirtualDirection(Location direction) {
        double yaw = direction.getYaw();
        yaw = yaw > 180 ?  yaw - 360 : yaw;
        if (yaw < -157.5) return BlockFace.NORTH;
        else if (yaw < -112.5) return BlockFace.NORTH_EAST;
        else if (yaw < -67.5) return BlockFace.EAST;
        else if (yaw < -22.5) return BlockFace.SOUTH_EAST;
        else if (yaw < 22.5) return BlockFace.SOUTH;
        else if (yaw < 67.5) return BlockFace.SOUTH_WEST;
        else if (yaw < 112.5) return BlockFace.WEST;
        else if (yaw < 157.5) return BlockFace.NORTH_WEST;
        else return BlockFace.NORTH;
    }

    public List<Block> getLastTwoTargets(Player ply, List<Material> ignored, int range) {
        BlockIterator iter = new BlockIterator(ply, Math.max(2 ,range));
        ArrayList<Block> lastTwo = new ArrayList<>();
        lastTwo.addAll(Arrays.asList(iter.next(), iter.next()));
        while (iter.hasNext()) {
            Block current = iter.next();
            lastTwo.set(1, lastTwo.get(0));
            lastTwo.set(0, current);
            if (!ignored.contains(current.getType())) break;
        }
        if (lastTwo.size() < 2) {
            Block head = ply.getEyeLocation().getBlock();
            return Arrays.asList(head, head.getRelative(getVirtualDirection(ply.getEyeLocation())));
        }
        return lastTwo;
    }
    public List<Block> getLastTwoTargets(Player ply, int range) {
        return getLastTwoTargets(ply, Arrays.asList(Material.AIR), range);
    }

    public Location getTargetPoint(Player ply, List<Material> ignored, int range) {
        List<Block> lastTwo = getLastTwoTargets(ply, ignored, range);
        BlockFace face = lastTwo.get(1).getFace(lastTwo.get(0));
        if (face == null) return null;
        Vector planePoint = lastTwo.get(0).getLocation().toVector();

        switch(face) {
            case EAST: planePoint.setX(planePoint.getX() + 1);
            case SOUTH: planePoint.setZ(planePoint.getZ() + 1);
            case UP: planePoint.setY(planePoint.getY() + 1);
        }

        Vector planeNormal = new Vector(face.getModX(), face.getModY(), face.getModZ());
        Vector linePointOne = ply.getEyeLocation().toVector();
        Vector linePointTwo = linePointOne.clone().add(ply.getLocation().getDirection());

        Vector finalPoint = lineIntersectPlane(linePointOne, linePointTwo, planePoint, planeNormal);
        return new Location(ply.getWorld(), finalPoint.getX(), finalPoint.getY(), finalPoint.getZ());
    }
    public Location getTargetPoint(Player ply, int range) {
        return getTargetPoint(ply, Arrays.asList(Material.AIR), range);
    }

    public Vector lineIntersectPlane(Vector linePointOne, Vector linePointTwo, Vector planePoint, Vector planeNormal, double epsilon) {
        Vector u = linePointTwo.clone().subtract(linePointOne);
        Vector w = linePointOne.clone().subtract(planePoint);
        double dot = planeNormal.dot(u);
        if (Math.abs(dot) > epsilon) {
            u.multiply(-planeNormal.dot(w) / dot);
            return linePointOne.clone().add(u);
        } else return null;
    }
    public Vector lineIntersectPlane(Vector linePointOne, Vector linePointTwo, Vector planePoint, Vector planeNormal) {
        return lineIntersectPlane(linePointOne, linePointTwo, planePoint, planeNormal, 0.0000001);
    }
}
