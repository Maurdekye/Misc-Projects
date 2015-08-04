package mainpack;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerCommandPreprocessEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.java.JavaPlugin;

public class EnchantAll extends JavaPlugin implements Listener {
    public void onEnable() {
        Bukkit.getPluginManager().registerEvents(this, this);
    }

    @EventHandler(priority = EventPriority.HIGHEST)
    public void PreCommand(PlayerCommandPreprocessEvent event) {
        Player sender = event.getPlayer();
        if (!event.getPlayer().hasPermission("enchantall.use")) return;
        String[] commargs = event.getMessage().split(" +");
        if (commargs.length < 2) return;
        if (commargs[0].equalsIgnoreCase("/enchant") && commargs[1].equalsIgnoreCase("all")) {
            event.setCancelled(true);
            ItemStack held = sender.getItemInHand();
            boolean enchanted = false;
            for (Enchantment ench : Enchantment.values()) {
                try {
                    held.addEnchantment(ench, ench.getMaxLevel());
                    enchanted = true;
                } catch (Exception ignored) {}
            }
            String goodname = held.getType().name().replaceAll("_", " ").toLowerCase();
            if (enchanted)
                sender.sendMessage(ChatColor.AQUA + "Fully enchanted your " + goodname);
            else
                sender.sendMessage(ChatColor.GRAY + "You can't enchant " + goodname + "!");
        }
    }
}
