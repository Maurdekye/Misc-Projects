package mainpack;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Material;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.inventory.ClickType;
import org.bukkit.event.inventory.InventoryClickEvent;
import org.bukkit.event.player.PlayerDropItemEvent;
import org.bukkit.inventory.Inventory;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.meta.ItemMeta;
import org.bukkit.plugin.Plugin;

import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

public class IngameConfig implements Listener {

    Plugin parent;
    HashMap<String, Integer> values = new HashMap<>();

    private void init(Plugin parent, Map<? extends String, ? extends Integer> values) throws Exception {
        this.parent = parent;
        if (values.size() > 6)
            throw new Exception("Inventory Size Limit is 6 rows");
        this.values.putAll(values);
        parent.getServer().getPluginManager().registerEvents(this, parent);
    }
    IngameConfig(Plugin parent, Map<? extends String, ? extends Integer> values) throws Exception {
        this.init(parent, values);
    }
    IngameConfig(Plugin parent, Collection<? extends String> values) throws Exception {
        HashMap<String, Integer> toUse = new HashMap<>();
        for (String item : values)
            toUse.put(item, 0);
        this.init(parent, toUse);
    }

    @EventHandler(priority = EventPriority.HIGHEST)
    private void Inv(InventoryClickEvent event) {
        if (!(event.getWhoClicked() instanceof Player)) return;
        Player ply = (Player) event.getWhoClicked();
        if (!event.getInventory().getName().equals("Menu" + ChatColor.RESET)) {}
        else if (event.getRawSlot() > event.getInventory().getSize() || event.getRawSlot() < 0) {}
        else if (event.getClick() != ClickType.LEFT) {}
        else {
            if (event.getCursor() != null && event.getCursor().getType() == Material.REDSTONE_TORCH_ON) {
                event.setCancelled(true);
                int distance = ((event.getRawSlot() % 9) - 4);
                int finval = 2^Math.abs(Math.max(distance-1, 0));
                finval = distance < 0 ? -finval : finval;
                System.out.println(finval);
                String key = event.getCursor().getItemMeta().getDisplayName().split(":")[0];
                values.put(key, values.get(key) + distance);
                this.showMenu(ply);
            } else if (event.getCursor() == null) {
                this.showMenu(ply);
            }
        }
    }

    @EventHandler
    private void Drop(PlayerDropItemEvent event) {
        ItemStack dropped = event.getItemDrop().getItemStack();
        ItemMeta meta = dropped.getItemMeta();
        if (meta == null || meta.getLore() == null || meta.getLore().size() == 0) return;
        else if (meta.getLore().get(0).equals(ChatColor.BLACK + "menuitem"))
            event.getItemDrop().remove();
    }

    public void showMenu(Player ply) {
        Inventory menu = Bukkit.createInventory(null, values.size()*9, "Menu" + ChatColor.RESET);
        int i=0;
        for (String key : values.keySet()) {
            i++;
            ItemStack itemToSet = new ItemStack(Material.REDSTONE_TORCH_ON);
            ItemMeta meta = itemToSet.getItemMeta();
            meta.setDisplayName(key + ": " + values.get(key));
            meta.setLore(Arrays.asList(ChatColor.BLACK + "menuitem"));
            itemToSet.setItemMeta(meta);
            menu.setItem(i*9 - 5, itemToSet);
        }
        ply.openInventory(menu);
    }

    public void setValue(String key, int value) {
        values.put(key, value);
    }

    public int getValue(String key) {
        return values.containsKey(key)? values.get(key) : 0;
    }

    public String[] getKeys() {
        return values.keySet().toArray(new String[values.size()]);
    }

}
