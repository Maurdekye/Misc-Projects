package mainpack;

import com.sk89q.worldedit.bukkit.WorldEditPlugin;
import com.sk89q.worldedit.bukkit.selections.Selection;
import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;
import org.bukkit.scheduler.BukkitRunnable;
import org.bukkit.util.Vector;

import java.util.*;

public class Rewind extends JavaPlugin {

    public long ticker = 0;

    public HashMap<Box, Recording> areaRecordings = new HashMap<>();
    public ArrayList<Recording> userRecordings = new ArrayList<>();
    public ArrayList<Recording> savedRecordings = new ArrayList<>();
    public ArrayList<Tracer> keyframes = new ArrayList<>();
    public HashMap<UUID, String> latest = new HashMap<>();

    public void onEnable() {
        Bukkit.getPluginManager().registerEvents(new Listen(this), this);

        new BukkitRunnable() {
            public void run() {
                ticker++;
                for (Tracer key : keyframes.toArray(new Tracer[keyframes.size()])) {
                    key.advance();
                    if (key.curframe == 0 && key.resetting)
                        keyframes.remove(key);
                }
            }
        }.runTaskTimer(this, 0, 1);
    }

    @Override
    public boolean onCommand(CommandSender sender, Command cmd, String label, String[] args) {
        Player ply = sender instanceof Player ? (Player) sender : null;

        // Record

        if (cmd.getName().equals("record")) {
            if (ply == null) {
                sender.sendMessage("You must be a player to record actions.");
            } else if (!ply.hasPermission("rewind.record")) {
                sender.sendMessage("You don't have permission to do that.");
            } else if (args.length == 0) return false;
            else if (args.length >= 2 && Arrays.asList("me", "area").contains(args[0].toLowerCase())) {
                String recname = args[1];
                if (args[0].equalsIgnoreCase("me")) {
                    if (!nameCheck(recname)) {
                        sender.sendMessage("That name has already been used.");
                    } else if (!ply.hasPermission("rewind.record.self")) {
                        sender.sendMessage("You don't have permission to do that.");
                    } else {
                        userRecordings.add(new Recording(recname, ticker, ply));
                    }
                } else if (args[0].equalsIgnoreCase("area")) {
                    WorldEditPlugin wep = (WorldEditPlugin) Bukkit.getPluginManager().getPlugin("WorldEdit");
                    if (wep == null) {
                        sender.sendMessage("You must have WorldEdit installed in order to record an area.");
                    } else {
                        Selection sel = wep.getSelection(ply);
                        if (sel == null) {
                            sender.sendMessage(ChatColor.RED + "Make a selection first.");
                        } else if (!nameCheck(recname)) {
                            sender.sendMessage("That name has already been used.");
                        } else if (!ply.hasPermission("rewind.record.area")) {
                            sender.sendMessage("You don't have permission to do that.");
                        } else {
                            Box recbox = new Box(sel.getMaximumPoint().toVector(), sel.getMaximumPoint().toVector());
                            areaRecordings.put(recbox, new Recording(recname, ticker, ply));
                        }
                    }
                }
            } else {
                String recname = args[0];
                if (!nameCheck(args[0])) {
                    sender.sendMessage("That name has already been used.");
                } else if (!ply.hasPermission("rewind.record.self")) {
                    sender.sendMessage("You don't have permission to do that.");
                } else {
                    userRecordings.add(new Recording(recname, ticker, ply));
                }
            }
        }

        // End Recording

        else if (cmd.getName().equals("endrec")) {
            if (ply == null) {
                sender.sendMessage("You must be a player to record actions.");
            } else if (!ply.hasPermission("rewind.record")) {
                sender.sendMessage("You don't have permission to do that.");
            } else if (args.length == 0) {
                ArrayList<String> ends = new ArrayList<>();
                for (Recording rec : userRecordings.toArray(new Recording[userRecordings.size()])) {
                    if (rec.owner.equals(ply.getUniqueId())) {
                        ends.add(rec.name);
                        userRecordings.remove(rec);
                        savedRecordings.add(rec);
                    }
                }
                for (Box b : areaRecordings.keySet()) {
                    Recording associated = areaRecordings.get(b);
                    if (associated.owner.equals(ply.getUniqueId())) {
                        ends.add(associated.name);
                        areaRecordings.remove(b);
                        savedRecordings.add(associated);
                    }
                }
                if (ends.size() > 0) {
                    sender.sendMessage(ChatColor.DARK_GRAY + "Stopped the following recordings:");
                    for (String name : ends) {
                        sender.sendMessage(ChatColor.GRAY + " - " + ChatColor.RESET + name);
                    }
                } else {
                    sender.sendMessage("You don't have any active recordings!");
                }
            } else {
                boolean removed = false;
                for (Recording rec : userRecordings.toArray(new Recording[userRecordings.size()])) {
                    if (rec.owner.equals(ply.getUniqueId()) && rec.name.equals(args[0])) {
                        removed = true;
                        userRecordings.remove(rec);
                        savedRecordings.add(rec);
                        latest.put(ply.getUniqueId(), rec.name);
                    }
                }
                if (!removed) {
                    for (Box b : areaRecordings.keySet()) {
                        Recording associated = areaRecordings.get(b);
                        if (associated.owner.equals(ply.getUniqueId()) && associated.name.equals(args[0])) {
                            removed = true;
                            areaRecordings.remove(b);
                            savedRecordings.add(associated);
                            latest.put(ply.getUniqueId(), associated.name);
                        }
                    }
                }
                if (!removed) {
                    sender.sendMessage("You don't have any active recordings by the name '-rec-'!".replaceAll("-rec-", args[0]));
                } else {
                    sender.sendMessage(ChatColor.DARK_GRAY + "Stopped recording '-rec-'.".replaceAll("-rec-", args[0]));
                }
            }
        }

        // Replay

        else if (cmd.getName().equals("replay")) {
            if (ply == null) {
                sender.sendMessage("You must be a player to replay recordings.");
            } else if (!ply.hasPermission("rewind.playback")) {
                sender.sendMessage("You don't have permission to do that.");
            } else if (args.length == 0) {
                if (!latest.containsKey(ply.getUniqueId()) || latest.get(ply.getUniqueId()) == null) {
                    sender.sendMessage("You don't have any recent recordings.");
                } else {
                    String lure = latest.get(ply.getUniqueId());
                    Recording fetch = recByName(lure, savedRecordings);
                    if (fetch == null) {
                        sender.sendMessage("Your latest saved recording, '-rec-', no longer exists.".replaceAll("-rec-", lure));
                    } else if (tracerByName(lure, keyframes) != null) {
                        sender.sendMessage("That recording is already playing.");
                    } else {
                        keyframes.add(new Tracer(fetch));
                    }
                }
            } else {
                String lure = args[0];
                Recording fetch = recByName(lure, savedRecordings);
                if (fetch == null) {
                    sender.sendMessage("Could not find a recording by the name, '-rec-'.".replaceAll("-rec-", lure));
                } else if (tracerByName(lure, keyframes) != null) {
                    sender.sendMessage("That recording is already running.");
                } else {
                    keyframes.add(new Tracer(fetch));
                }
            }
        }

        // Pause

        else if (cmd.getName().equals("playpause")) {
            if (ply == null) {
                sender.sendMessage("You must be a player to pause replays.");
            } else if (!ply.hasPermission("rewind.playback.pause")) {
                sender.sendMessage("You don't have permission to do that.");
            } else if (args.length == 0) {
                int paused = 0;
                for (Tracer key : keyframes) {
                    if (!key.paused && key.parent.owner.equals(ply.getUniqueId())) {
                        key.paused = true;
                        paused++;
                    }
                }
                if (paused > 0) {
                    sender.sendMessage("Paused -num- running recordings.".replaceAll("-num-", "" + paused));
                } else {
                    for (Tracer key : keyframes) {
                        if (key.paused && key.parent.owner.equals(ply.getUniqueId())) {
                            key.paused = false;
                        }
                    }
                    sender.sendMessage("Playing all running recordings.");
                }
            } else {
                String lure = args[0];
                Tracer fetch = tracerByName(lure, keyframes);
                if (fetch == null) {
                    sender.sendMessage(ChatColor.GRAY + "That recording doesn't exist / isn't running right now.");
                } else {
                    if (fetch.paused) {
                        fetch.paused = false;
                        sender.sendMessage("Playing recording '-rec-'.".replaceAll("-rec-", fetch.parent.name));
                    } else {
                        fetch.paused = true;
                        sender.sendMessage("Paused recording '-rec-'.".replaceAll("-rec-", fetch.parent.name));
                    }
                }
            }
        }

        // Stop

        else if (cmd.getName().equals("playstop")) {
            if (ply == null) {
                sender.sendMessage("You must be a player to stop replays.");
            } else if (!ply.hasPermission("rewind.playback.stop")) {
                sender.sendMessage("You don't have permission to do that.");
            } else {
                boolean reset = false;
                List<String> arglist = Arrays.asList(args);
                if (arglist.contains("-r") && ply.hasPermission("rewind.playback.stop.reset")) {
                    reset = true;
                    arglist.remove("-r");
                    args = arglist.toArray(new String[args.length]);
                }
                if (args.length == 0) {
                    int count = 0;
                    for (Tracer key : keyframes.toArray(new Tracer[keyframes.size()])) {
                        if (!key.resetting && key.parent.owner.equals(ply.getUniqueId())) {
                            if (reset) {
                                key.resetting = true;
                            } else {
                                keyframes.remove(key);
                            }
                            count++;
                        }
                    }
                    sender.sendMessage("Stopped -num- recordings.".replaceAll("-num-", "" + count));
                } else {
                    String lure = args[0];
                    Tracer fetch = tracerByName(lure, keyframes);
                    if (fetch == null) {
                        sender.sendMessage(ChatColor.GRAY + "That recording doesn't exist / isn't running right now.");
                    } else {
                        if (reset) {
                            fetch.resetting = true;
                        } else {
                            keyframes.remove(fetch);
                        }
                    }
                }
            }
        }

        // Rewind

        else if (cmd.getName().equals("rewind")) {
            if (ply == null) {
                sender.sendMessage("You must be a player to rewind replays.");
            } else if (!ply.hasPermission("rewind.playback.rewind")) {
                sender.sendMessage("You don't have permission to do that.");
            } else if (args.length == 0) {
                int rewound = 0;
                for (Tracer key : keyframes) {
                    if (!key.rewind && key.parent.owner.equals(ply.getUniqueId())) {
                        key.rewind = true;
                        rewound++;
                    }
                }
                if (rewound > 0) {
                    sender.sendMessage("Rewinding -num- running recordings.".replaceAll("-num-", "" + rewound));
                } else {
                    for (Tracer key : keyframes) {
                        if (key.rewind && key.parent.owner.equals(ply.getUniqueId())) {
                            key.rewind = false;
                        }
                    }
                    sender.sendMessage("Playing forward all running recordings.");
                }
            } else {
                String lure = args[0];
                Tracer fetch = tracerByName(lure, keyframes);
                if (fetch == null) {
                    sender.sendMessage(ChatColor.GRAY + "That recording doesn't exist / isn't running right now.");
                } else {
                    if (fetch.rewind) {
                        fetch.rewind = false;
                        sender.sendMessage("Playing forward recording '-rec-'.".replaceAll("-rec-", fetch.parent.name));
                    } else {
                        fetch.rewind = true;
                        sender.sendMessage("Rewinding recording '-rec-'.".replaceAll("-rec-", fetch.parent.name));
                    }
                }
            }
        }

        else if (cmd.getName().equals("playspeed")) {
            if (ply == null) {
                sender.sendMessage("You must be a player to speed up replays.");
            } else if (!ply.hasPermission("rewind.playback.speed")) {
                sender.sendMessage("You don't have permission to do that.");
            } else if (args.length == 0) return false;
            else {
                Integer speed;
                try {
                    speed = Integer.parseInt(args[Math.max(args.length, 2)-1]);
                } catch (NumberFormatException ignored) {
                    return false;
                }

                if (args.length == 1) {
                    int spedup = 0;
                    for (Tracer key : keyframes) {
                        if (key.speed != speed && key.parent.owner.equals(ply.getUniqueId())) {
                            key.speed = speed;
                            spedup++;
                        }
                    }
                    sender.sendMessage("Changed speed of -num- recordings to -spd-.".replaceAll("-num-", "" + spedup).replaceAll("-spd-", "" + speed));
                } else {
                    String lure = args[0];
                    Tracer fetch = tracerByName(lure, keyframes);
                    if (fetch == null) {
                        sender.sendMessage(ChatColor.GRAY + "That recording doesn't exist / isn't running right now.");
                    } else {
                        fetch.speed = speed;
                        sender.sendMessage("Adjusted the speed of recording -rec- to -spd-.".replaceAll("-rec-", fetch.parent.name).replaceAll("-spd-", "" + speed));
                    }
                }
            }
        }

        return true;
    }

    public boolean nameCheck(String name) {
        for (Recording rec : areaRecordings.values())
            if (rec.name.equals(name)) return false;
        for (Recording rec : userRecordings)
            if (rec.name.equals(name)) return false;
        return true;
    }

    /*@SafeVarargs
    public final <T> ArrayList<T> concatenate(Collection<T>... lists) {
        ArrayList<T> finlist = new ArrayList<>();
        for (Collection<T> list : lists)
            finlist.addAll(list);
        return finlist;
    }*/

    public Recording recByName(String name, Collection<Recording> from) {
        for (Recording rec : from)
            if (rec.name.equals(name)) return rec;
        return null;
    }

    public Tracer tracerByName(String name, Collection<Tracer> from) {
        for (Tracer keyframe : from)
            if (keyframe.parent.name.equals(name) && !keyframe.resetting) return keyframe;
        return null;
    }

}

class Box {

    Vector min, max;
    public Box(Vector min, Vector max) {
        this.min = min;
        this.max = max;

        if (min.getX() > max.getX()) {
            double placeholder = min.getX();
            min.setX(max.getX());
            max.setX(placeholder);
        }

        if (min.getY() > max.getY()) {
            double placeholder = min.getY();
            min.setY(max.getY());
            max.setY(placeholder);
        }

        if (min.getZ() > max.getZ()) {
            double placeholder = min.getZ();
            min.setZ(max.getZ());
            max.setZ(placeholder);
        }
    }

    public boolean contains(Vector check) {
        return check.getX() <= max.getX() &&
                check.getX() >= min.getX() &&
                check.getY() <= max.getY() &&
                check.getY() >= min.getY() &&
                check.getZ() <= max.getZ() &&
                check.getZ() >= min.getZ();
    }
}