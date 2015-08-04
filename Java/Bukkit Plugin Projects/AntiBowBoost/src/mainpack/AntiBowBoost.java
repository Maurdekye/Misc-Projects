package mainpack;

import org.bukkit.Bukkit;
import org.bukkit.entity.Arrow;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.plugin.java.JavaPlugin;

public class AntiBowBoost extends JavaPlugin implements Listener {
    public void onEnable() {
        Bukkit.getPluginManager().registerEvents(this, this);
    }

    @EventHandler
    public void EntDmgEnt(EntityDamageByEntityEvent event) {
        Entity damager = event.getDamager();
        Entity victim = event.getEntity();
        if (damager instanceof Arrow && victim instanceof Player) {
            Arrow arrow = (Arrow) damager;
            Player ply = (Player) victim;
            if (arrow.getShooter().equals(ply) && !ply.hasPermission("anti.boost")) {
                event.setCancelled(true);
                ply.damage(event.getDamage());
            }
        }
    }
}
