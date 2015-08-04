package mainpack;

import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.block.Action;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.ShapedRecipe;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.java.JavaPlugin;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LandMine extends JavaPlugin implements Listener {

    CustomConfig mineCfg = new CustomConfig(this, "mines.yml");

    @Override
    public void onEnable() {
        Bukkit.getPluginManager().registerEvents(this, this);

        mineCfg.getConfig().addDefault("active", true);
        mineCfg.getConfig().addDefault("power", 4);
        mineCfg.getConfig().options().copyDefaults(true);
        mineCfg.saveConfig();

        ItemStack plateItem = new ItemStack(Material.STONE_PLATE);
        ItemMeta meta = plateItem.getItemMeta();
        meta.setLore(Arrays.asList("Landmine"));
        plateItem.setItemMeta(meta);
        ShapedRecipe plateRecipe = new ShapedRecipe(plateItem);
        plateRecipe.shape("+#+").setIngredient('+', Material.STONE).setIngredient('#', Material.TNT);
        Bukkit.addRecipe(plateRecipe);
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {

        if (command.getName().equalsIgnoreCase("landmine")) {
            if (args.length == 0 || !Arrays.asList("on", "off").contains(args[0].toLowerCase())) return false;
            if (sender instanceof Player && !sender.hasPermission("landmine.toggle")) {
                sender.sendMessage("You're not allowed to use that command!");
            } else {
                if (args[0].equalsIgnoreCase("on")) {
                    mineCfg.getConfig().set("active", true);
                    sender.sendMessage("Activated landmines.");
                } else if (args[0].equalsIgnoreCase("off")) {
                    mineCfg.getConfig().set("active", false);
                    sender.sendMessage("Deactivated landmines.");
                }
                mineCfg.saveConfig();
            }
        }

        return true;
    }

    @EventHandler
    public void Place(BlockPlaceEvent event) {
        if (!event.getPlayer().hasPermission("landmine.use")) return;
        if (event.getBlock().getType() != Material.STONE_PLATE) return;
        List<String> lore = event.getItemInHand().getItemMeta().getLore();
        if (lore == null || !lore.get(0).equals("Landmine")) return;

        ArrayList<Location> mines = extractLocations(mineCfg.getConfig().getStringList("mines"));
        mines.add(event.getBlock().getLocation());
        mineCfg.getConfig().set("mines", hashLocations(mines));
        mineCfg.saveConfig();
    }

    @EventHandler
    public void Break(BlockBreakEvent event) {
        if (event.getBlock().getType() != Material.STONE_PLATE) return;
        ArrayList<Location> mines = extractLocations(mineCfg.getConfig().getStringList("mines"));
        if (!mines.contains(event.getBlock().getLocation())) return;

        mines.remove(event.getBlock().getLocation());
        mineCfg.getConfig().set("mines", hashLocations(mines));
        mineCfg.saveConfig();
    }

    @EventHandler
    public void Interact(PlayerInteractEvent event) {
        if (event.getAction() != Action.PHYSICAL) return;
        if (event.getClickedBlock().getType() != Material.STONE_PLATE) return;
        if (!mineCfg.getConfig().getBoolean("active")) return;
        ArrayList<Location> mines = extractLocations(mineCfg.getConfig().getStringList("mines"));
        if (!mines.contains(event.getClickedBlock().getLocation())) return;

        mines.remove(event.getClickedBlock().getLocation());
        mineCfg.getConfig().set("mines", hashLocations(mines));
        mineCfg.saveConfig();
        final Block finalBlock = event.getClickedBlock();
        Bukkit.getScheduler().scheduleSyncDelayedTask(this, new Runnable() {
            public void run() {
                finalBlock.getWorld().createExplosion(finalBlock.getLocation(), mineCfg.getConfig().getInt("power"));
            }
        }, 10);
    }

    public ArrayList<Location> extractLocations(List<String> hash) {
        ArrayList<Location> loclist = new ArrayList<>();
        for (String locstr : hash) {
            String[] pieces = locstr.split("\\|");
            World grabWorld = Bukkit.getWorld(pieces[3]);
            if (grabWorld == null) grabWorld = Bukkit.getWorlds().get(0);
            loclist.add(new Location(grabWorld,
                    Integer.parseInt(pieces[0]),
                    Integer.parseInt(pieces[1]),
                    Integer.parseInt(pieces[2])));
        }
        return loclist;
    }

    public List<String> hashLocations(ArrayList<Location> locations) {
        ArrayList<String> hash = new ArrayList<>();
        for (Location l : locations)
            hash.add(l.getBlockX() + "|" + l.getBlockY() + "|" + l.getBlockZ() + "|" + l.getWorld().getName());
        return hash;
    }
}
