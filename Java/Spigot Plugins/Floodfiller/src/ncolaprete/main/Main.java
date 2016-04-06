package ncolaprete.main;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import org.bukkit.ChatColor;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.block.BlockFace;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.util.BlockIterator;

public class Main extends org.bukkit.plugin.java.JavaPlugin
{
    public HashMap<Player, ArrayList<HashMap<int[], Material>>> undos = new HashMap();
    public int defLimit = 5000;
    public int maxLimit = 20000;
    public int timeLimit = 8;
    
    public void onEnable() {
        getConfig().options().copyDefaults(true);
        saveConfig();
        this.defLimit = getConfig().getInt("default_limit");
        this.maxLimit = getConfig().getInt("maximum_limit");
        this.timeLimit = getConfig().getInt("timeout_limit");
    }
    
    public boolean eq(String arg1, String arg2) {
        if (arg1.equalsIgnoreCase(arg2)) {
            return true;
        }
        return false;
    }
    
    public void listPlayer(Player ply)
    {
        if (!this.undos.containsKey(ply))
            this.undos.put(ply, new ArrayList());
    }
    
    public boolean listHasItem(String[] list, String item) {
        String[] arrayOfString;
        int j = (arrayOfString = list).length; for (int i = 0; i < j; i++) { String e = arrayOfString[i];
            if (e.equalsIgnoreCase(item)) {
                return true;
            }
        }
        return false;
    }
    
    public boolean isLegalOP(CommandSender sndr) {
        if ((sndr instanceof Player)) {
            Player plr = (Player)sndr;
            if (plr.isOp()) {
                return true;
            }
            plr.sendMessage(ChatColor.RED + "Sorry, you can't use that command.");
        }
        
        return false;
    }
    
    public boolean isLegalPLY(CommandSender sndr) {
        if ((sndr instanceof Player)) {
            return true;
        }
        return false;
    }
    
    public boolean isLegalCMD(CommandSender sndr) {
        if ((sndr instanceof Player)) {
            Player plr = (Player)sndr;
            if (plr.isOp()) {
                return true;
            }
            plr.sendMessage(ChatColor.RED + "Sorry, you can't use that command.");
        }
        
        return true;
    }
    
    public int[] blockToIntList(Block block) {
        return new int[] { block.getX(), block.getY(), block.getZ() };
    }
    
    public void undo(Player ply, int amount) {
        if (amount > ((ArrayList)this.undos.get(ply)).size()) amount = ((ArrayList)this.undos.get(ply)).size();
        for (int i = 0; i < amount; i++) {
            int last = ((ArrayList)this.undos.get(ply)).size() - 1;
            for (Object eo : ((HashMap)((ArrayList)this.undos.get(ply)).get(last)).keySet()) {
                int[] e = (int[]) eo;
                ply.getWorld().getBlockAt(e[0], e[1], e[2]).setType((Material)((HashMap)((ArrayList)this.undos.get(ply)).get(last)).get(e));
            }
            ((ArrayList)this.undos.get(ply)).remove(last);
        }
    }
    
    public boolean isBlock(World wrld, Material id) {
        Material cid = wrld.getBlockAt(0, 0, 0).getType();
        try {
            wrld.getBlockAt(0, 0, 0).setType(id);
        } catch (NullPointerException e) {
            return false;
        }
        wrld.getBlockAt(0, 0, 0).setType(cid);
        return true;
    }
    
    public Block getTarget(Player ply, int limit) {
        BlockIterator iter = new BlockIterator(ply, limit);
        Block cur = null;
        while (iter.hasNext()) {
            cur = iter.next();
            if (cur.getType() != Material.AIR)
                break;
        }
        return cur;
    }
    
    public static Block getOuter(Player ply, int range) {
        BlockIterator sight = new BlockIterator(ply, range);
        Block lastblock = null;
        while (sight.hasNext()) {
            Block curblock = sight.next();
            if (curblock.getType() == Material.AIR) {
                lastblock = curblock;
            }
            else {
                return lastblock;
            }
        }
        return null;
    }
    
    public static Block getInner(Player ply, int range) {
        BlockIterator sight = new BlockIterator(ply, range);
        Block lastblock = null;
        while (sight.hasNext()) {
            Block curblock = sight.next();
            if (curblock.getType() == Material.AIR) {
                lastblock = curblock;
            }
            else {
                BlockFace f = lastblock.getFace(curblock);
                return curblock.getRelative(f);
            }
        }
        return null;
    }
    
    private boolean deepContains(ArrayList<int[]> check, int[] item) {
        for (int[] e : check) {
            boolean isIn = true;
            if (e.length != item.length) {
                isIn = false;
            } else {
                for (int i = 0; i < e.length; i++) {
                    if (e[i] != item[i])
                        isIn = false;
                }
            }
            if (isIn)
                return true;
        }
        return false;
    }
    
    private ArrayList<int[]> getSurrounds(int[] center, boolean shallow) {
        int x = center[0];
        int y = center[1];
        int z = center[2];
        ArrayList<int[]> toCheck = new ArrayList();
        if (shallow) {
            toCheck.add(new int[] { x + 1, y, z });
            toCheck.add(new int[] { x - 1, y, z });
            toCheck.add(new int[] { x, y + 1, z });
            toCheck.add(new int[] { x, y - 1, z });
            toCheck.add(new int[] { x, y, z + 1 });
            toCheck.add(new int[] { x, y, z - 1 });
        } else {
            for (int xi = x - 1; xi <= x + 1; xi++) {
                for (int yi = y - 1; yi <= y + 1; yi++) {
                    for (int zi = z - 1; zi <= z + 1; zi++) {
                        int[] assign = { xi, yi, zi };
                        toCheck.add(assign);
                        if ((xi == x) && (yi == y) && (zi == z)) toCheck.remove(assign);
                    }
                }
            }
        }
        return toCheck;
    }
    



    public void floodfill(Player ply, World wrld, int[] start, Material get, byte getData, Material set, byte setData, int lim, int count, boolean diag, long startTime)
    {
        int x = start[0];
        int y = start[1];
        int z = start[2];
        if ((y < 0) || (y > 256)) return;
        if (wrld.getBlockAt(x, y, z).getType() != get) return;
        if (count >= lim) return;
        if (System.currentTimeMillis() > startTime + this.timeLimit * 1000) return;
        ((HashMap)((ArrayList)this.undos.get(ply)).get(((ArrayList)this.undos.get(ply)).size() - 1)).put(new int[] { x, y, z }, wrld.getBlockAt(x, y, z).getType());
        wrld.getBlockAt(x, y, z).setType(set);
        wrld.getBlockAt(x, y, z).setData(setData);
        try { int[] c;
            for (Iterator localIterator = getSurrounds(start, !diag).iterator(); localIterator.hasNext(); floodfill(ply, wrld, c, get, getData, set, setData, lim, count + 1, diag, startTime)) { c = (int[])localIterator.next();
            }
        }
        catch (Throwable e) {}
    }
    

    public void hollow(Player ply, World wrld, int x, int y, int z, Material get, Material set, byte getData, int lim, boolean diag, long startTime)
    {
        for (int[] item : hollowList(wrld, new int[] { x, y, z }, get, lim, 0, new ArrayList(), new ArrayList(), diag, startTime)) {
            ((HashMap)((ArrayList)this.undos.get(ply)).get(((ArrayList)this.undos.get(ply)).size() - 1)).put(new int[] { item[0], item[1], item[2] }, wrld.getBlockAt(item[0], item[1], item[2]).getType());
            wrld.getBlockAt(item[0], item[1], item[2]).setType(set);
            wrld.getBlockAt(item[0], item[1], item[2]).setData(getData);
        }
    }
    
    private ArrayList<int[]> hollowList(World wrld, int[] start, Material get, int lim, int count, ArrayList<int[]> list, ArrayList<int[]> checked, boolean diag, long startTime) { int x = start[0];
        int y = start[1];
        int z = start[2];
        if (deepContains(checked, new int[] { x, y, z })) return list;
        checked.add(new int[] { x, y, z });
        if ((y < 0) || (y > 256)) return list;
        if (wrld.getBlockAt(x, y, z).getType() != get) return list;
        if (count >= lim) return list;
        if (System.currentTimeMillis() > startTime + this.timeLimit * 1000) return list;
        for (int[] c : getSurrounds(start, !diag)) {
            if (wrld.getBlockAt(c[0], c[1], c[2]).getType() != get) return list;
        }
        list.add(start);
        try {
            int[] c;
            for (Iterator<int[]> i = getSurrounds(start, !diag).iterator(); i.hasNext(); list = hollowList(wrld, c, get, lim, count + 1, list, checked, diag, startTime)) c = (int[])i.next();
        } catch (Throwable e) {
            return list;
        }
        return list;
    }
    

    public void shell(Player ply, World wrld, int x, int y, int z, Material get, byte getData, Material set, int lim, boolean over, long startTime)
    {
        for (int[] item : shellList(wrld, new int[] { x, y, z }, get, lim, 0, new ArrayList(), new ArrayList(), over, startTime)) {
            ((HashMap)((ArrayList)this.undos.get(ply)).get(((ArrayList)this.undos.get(ply)).size() - 1)).put(new int[] { item[0], item[1], item[2] }, wrld.getBlockAt(item[0], item[1], item[2]).getType());
            wrld.getBlockAt(item[0], item[1], item[2]).setType(set);
            wrld.getBlockAt(item[0], item[1], item[2]).setData(getData);
        }
    }
    
    private ArrayList<int[]> shellList(World wrld, int[] start, Material get, int lim, int count, ArrayList<int[]> changes, ArrayList<int[]> checks, boolean over, long startTime) { int x = start[0];
        int y = start[1];
        int z = start[2];
        if ((y > 256) || (y < 0)) return changes;
        if (count >= lim) return changes;
        if (deepContains(checks, start)) return changes;
        checks.add(start);
        if (wrld.getBlockAt(x, y, z).getType() != get) return changes;
        if (System.currentTimeMillis() > startTime + this.timeLimit * 1000) return changes;
        boolean left = true;
        for (int[] c : getSurrounds(start, false)) {
            if (wrld.getBlockAt(c[0], c[1], c[2]).getType() != get) {
                left = false;
                break;
            }
        }
        if (left) return changes;
        for (int[] c : getSurrounds(start, false)) {
            if (wrld.getBlockAt(c[0], c[1], c[2]).getType() != get)
                if ((over) && (wrld.getBlockAt(c[0], c[1], c[2]).getType() == Material.AIR)) {
                    if (!deepContains(changes, c)) changes.add(c);
                }
                else if ((!over) && (!deepContains(changes, c))) changes.add(c);
        }
        try {
            int[] c;
            for (Iterator<int[]> i = getSurrounds(start, false).iterator(); i.hasNext(); changes = shellList(wrld, c, get, lim, count + 1, changes, checks, over, startTime)) c = (int[])i.next();
        } catch (Throwable e) {
            return changes;
        }
        return changes;
    }
    

    public void betterShell(Player ply, World wrld, int x, int y, int z, Material get, byte getData, Material set, int lim, boolean checkHeavy, boolean moveHeavy, boolean overwrite, long startTime)
    {
        for (int[] item : betterShellList(wrld, new int[] { x, y, z }, get, lim, 0, new ArrayList(), new ArrayList(), checkHeavy, moveHeavy, overwrite, startTime)) {
            ((HashMap)((ArrayList)this.undos.get(ply)).get(((ArrayList)this.undos.get(ply)).size() - 1)).put(new int[] { item[0], item[1], item[2] }, wrld.getBlockAt(item[0], item[1], item[2]).getType());
            wrld.getBlockAt(item[0], item[1], item[2]).setType(set);
            wrld.getBlockAt(item[0], item[1], item[2]).setData(getData);
        }
    }
    
    private ArrayList<int[]> betterShellList(World wrld, int[] start, Material get, int lim, int depth, ArrayList<int[]> checked, ArrayList<int[]> change, boolean checkheavy, boolean moveheavy, boolean overwrite, long startTime) { int x = start[0];
        int y = start[1];
        int z = start[2];
        if ((y > 255) || (y < 1)) return change;
        if (deepContains(checked, start)) return change;
        checked.add(start);
        if (depth >= lim) return change;
        if (wrld.getBlockAt(x, y, z).getType() == get) return change;
        if ((overwrite) && (wrld.getBlockAt(x, y, z).getType() != Material.AIR)) return change;
        if (System.currentTimeMillis() > startTime + this.timeLimit * 1000) return change;
        boolean found = false;
        for (int[] c : getSurrounds(start, checkheavy))
            if (wrld.getBlockAt(c[0], c[1], c[2]).getType() == get) found = true;
        if (!found) { return change;
        }
        change.add(start);
        try { int[] c;
            for (Iterator<int[]> i = getSurrounds(start, moveheavy).iterator(); i.hasNext(); change = betterShellList(wrld, c, get, lim, depth + 1, checked, change, checkheavy, moveheavy, overwrite, startTime)) c = (int[])i.next();
        } catch (Throwable e) {
            return change;
        }
        return change;
    }
    
    public String printList(int[] inlis) {
        String to_sender = "{";
        for (int i = 0; i < inlis.length - 1; i++) {
            to_sender = to_sender + inlis[i] + ", ";
        }
        return to_sender + inlis[(inlis.length - 1)] + "}";
    }
    
    public String[] extract(String[] command, String flag) {
        ArrayList<String> ret = new ArrayList();
        String[] arrayOfString1; int j = (arrayOfString1 = command).length; for (int i = 0; i < j; i++) { String s = arrayOfString1[i];
            if (!s.equals(flag)) ret.add(s);
        }
        String[] fin = new String[ret.size()];
        for (int i = 0; i < ret.size(); i++) {
            fin[i] = ((String)ret.get(i));
        }
        return fin;
    }


    public boolean onCommand(CommandSender sender, Command cmd, String cname, String[] args)
    {
        long curTime = System.currentTimeMillis();
        Player ply = null;
        if ((sender instanceof Player)) ply = (Player)sender;
        listPlayer(ply);
        

        byte data = 0;
        String grabId = "0";
        String grabData = "0";
        String[] flagnames = { "-d", "-o", "-m", "-c" };
        HashMap<String, Boolean> flags = new HashMap();
        String[] arrayOfString1; int j = (arrayOfString1 = flagnames).length; for (int i = 0; i < j; i++) { String flag = arrayOfString1[i];
            if (listHasItem(args, flag)) {
                flags.put(flag, Boolean.valueOf(true));
                args = extract(args, flag);
            } else {
                flags.put(flag, Boolean.valueOf(false));
            }
        }
        if (args.length > 0) {
            grabId = args[0];
            for (int i = 0; i < args[0].length(); i++)
                if (args[0].substring(i, i + 1).equals(":")) {
                    grabId = args[0].substring(0, i);
                    grabData = args[0].substring(i + 1, args[0].length());
                }
        }
        try {
            int limit;
            if (args.length < 2) limit = this.defLimit; else
                limit = Integer.parseInt(args[1]);
            data = Byte.parseByte(grabData);
        } catch (NumberFormatException e) {
            sender.sendMessage(ChatColor.RED + "Arguments must be integers.");
            return false; }
        int limit = 0;
        Material id;
        try { id = Material.getMaterial(Integer.parseInt(grabId));
        } catch (NumberFormatException e) {
            id = Material.getMaterial(grabId.toUpperCase());
        }
        if (!isBlock(ply.getWorld(), id)) {
            sender.sendMessage(ChatColor.RED + "Not a valid block: " + id);
            return false;
        }
        if (limit > this.maxLimit) {
            limit = this.maxLimit;
            sender.sendMessage(ChatColor.RED + "Hard limit is " + this.maxLimit + ", lowering to that.");
        }
        Block lookBlock = getTarget(ply, 100);
        Block inBlock = getInner(ply, 100);
        Block outBlock = getOuter(ply, 100);
        int lookDat = lookBlock.getData();
        Material look = lookBlock.getType();
        int x = lookBlock.getX();
        int y = lookBlock.getY();
        int z = lookBlock.getZ();
        int[] ins = { inBlock.getX(), inBlock.getY(), inBlock.getZ() };
        int[] outs = { outBlock.getX(), outBlock.getY(), outBlock.getZ() };
        



        if ((eq(cname, "flood")) && (isLegalOP(sender))) {
            if (args.length == 0) {
                sender.sendMessage(ChatColor.RED + "Usage: /flood <block id> [limit] [-d]");
            } else {
                if ((id == look) && (data == lookDat)) {
                    sender.sendMessage(ChatColor.RED + "Cannot flood a structure with its own block.");
                    return false;
                }
                sender.sendMessage(ChatColor.AQUA + "Flooding blocks with " + id + ", data " + data + "..");
                ((ArrayList)this.undos.get(ply)).add(new HashMap());
                floodfill(ply, ply.getWorld(), blockToIntList(lookBlock), look, (byte)lookDat, id, data, limit, 0, ((Boolean)flags.get("-d")).booleanValue(), curTime);
                sender.sendMessage(ChatColor.AQUA + "Successfully flood filled the object.");
                return true;
            }
        }
        else {
            if ((eq(cname, "hollow")) && (isLegalOP(sender))) {
                if (id == Material.AIR) sender.sendMessage(ChatColor.AQUA + "Hollowing out the object..");
                sender.sendMessage(ChatColor.AQUA + "Hollowing out structure with " + id + ", data " + data + "..");
                ((ArrayList)this.undos.get(ply)).add(new HashMap());
                hollow(ply, ply.getWorld(), ins[0], ins[1], ins[2], look, id, data, limit, ((Boolean)flags.get("-d")).booleanValue(), curTime);
                if (id == Material.AIR) sender.sendMessage(ChatColor.AQUA + "Successfully hollowed out the object."); else
                    sender.sendMessage(ChatColor.AQUA + "Successfully filled up the object.");
                return true;
            }
            
            if ((eq(cname, "floodout")) && (isLegalOP(sender))) {
                if (args.length == 0) {
                    sender.sendMessage(ChatColor.RED + "Usage: /floodout <block id> [limit]");
                } else {
                    lookBlock = ply.getLocation().getBlock();
                    lookDat = lookBlock.getData();
                    look = lookBlock.getType();
                    x = lookBlock.getX();
                    y = lookBlock.getY();
                    z = lookBlock.getZ();
                    if (id == look) {
                        sender.sendMessage(ChatColor.RED + "Cannot flood a fluid with more of itself.");
                        return false;
                    }
                    sender.sendMessage(ChatColor.AQUA + "Flooding blocks with " + id + ", data " + data + "..");
                    ((ArrayList)this.undos.get(ply)).add(new HashMap());
                    floodfill(ply, ply.getWorld(), new int[] { x, y, z }, look, (byte)lookDat, id, data, limit, 0, ((Boolean)flags.get("-d")).booleanValue(), curTime);
                    sender.sendMessage(ChatColor.AQUA + "Successfully flood filled the object.");
                    return true;
                }
                
            }
            else if ((eq(cname, "shell")) && (isLegalOP(sender))) {
                if (args.length == 0) {
                    sender.sendMessage(ChatColor.RED + "Usage: /shell <block id> [limit] [-c] [-m] [-o]");
                } else {
                    sender.sendMessage(ChatColor.AQUA + "Coating structure with " + id + ", data " + data + "..");
                    ((ArrayList)this.undos.get(ply)).add(new HashMap());
                    betterShell(ply, ply.getWorld(), outs[0], outs[1], outs[2], look, data, id, limit, ((Boolean)flags.get("-c")).booleanValue(), ((Boolean)flags.get("-m")).booleanValue(), !((Boolean)flags.get("-o")).booleanValue(), curTime);
                    sender.sendMessage(ChatColor.AQUA + "Successfully coated the object.");
                    return true;
                }
                
            }
            else if ((eq(cname, "envelop")) && (isLegalOP(sender))) {
                if (args.length == 0) {
                    sender.sendMessage(ChatColor.RED + "Usage: /envelop <block id> [limit] [-o]");
                } else {
                    sender.sendMessage(ChatColor.AQUA + "Coating structure with " + id + ", data " + data + "..");
                    ((ArrayList)this.undos.get(ply)).add(new HashMap());
                    shell(ply, ply.getWorld(), x, y, z, look, data, id, limit, !((Boolean)flags.get("-o")).booleanValue(), curTime);
                    sender.sendMessage(ChatColor.AQUA + "Successfully coated the object.");
                    return true;
                }
                
            }
            else if ((eq(cname, "undofill")) && (isLegalOP(sender))) {
                int amount = 1;
                try {
                    if (args.length >= 1) amount = Math.abs(Integer.parseInt(args[0]));
                } catch (NumberFormatException localNumberFormatException1) {}
                if (((ArrayList)this.undos.get(ply)).size() == 0) {
                    sender.sendMessage(ChatColor.RED + "You don't have any actions left to undo.");
                } else {
                    undo(ply, amount);
                    if (amount == 1) sender.sendMessage(ChatColor.AQUA + "Successfully undid last action."); else
                        sender.sendMessage(ChatColor.AQUA + "Successfully undid last " + amount + " actions.");
                    return true;
                }
            } }
        return false;
    }
}