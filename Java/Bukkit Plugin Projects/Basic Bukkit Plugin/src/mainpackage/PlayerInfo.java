package mainpackage;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Scanner;

import org.bukkit.entity.Player;

/*
 * Players should be registered upon joining and when the plugin is activated
 * Player info should be saved upon their leave and when the plugin is deactivated
 * Preferably have commands for these to manually save or register players
 */

class PlayerInfo {
	public HashMap<Player, HashMap<String, Integer>> info = new HashMap<Player, HashMap<String, Integer>>();
	public String[] values;
	public File save_folder;
	
	PlayerInfo(String[] values, String save_path) {
		this.values = values;
		this.save_folder = new File(save_path);
		if (!this.save_folder.exists()) {
			this.save_folder.mkdir();
		}
	}
	
	public boolean register(Player ply) {
		if (this.info.containsKey(ply)) {
			return false;
		}
		File plyfile = null;
		for (File f : this.save_folder.listFiles()) {
			String n = f.getName();
			if (ply.getName().equalsIgnoreCase(n.substring(0, n.length()-6))) {
				plyfile = f;
				break;
			}
		}
		HashMap<String, Integer> toAdd = new HashMap<String, Integer>();
		if  (plyfile == null) {
			for (String val : this.values) {
				toAdd.put(val, 0);
			}
			this.save(ply);
		} else {
			try {
				Scanner scan = new Scanner(plyfile);
				while (scan.hasNextLine()) {
					String[] vals = scan.nextLine().split(":");
					if (hasString(this.values, vals[0])) {
						toAdd.put(vals[0], Integer.parseInt(vals[1]));
					}
				}
				scan.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
				return false;
			}
		}
		this.info.put(ply, toAdd);
		return true;
	}

	public boolean setValue(Player ply, String key, int value) {
		this.register(ply);
		if (!hasString(this.values, key)) {return false;}
		HashMap<String, Integer> toAdd = new HashMap<String, Integer>();
		toAdd.put(key, value);	
		this.info.put(ply, toAdd);
		return true;
	}
	
	public HashMap<String, Integer> getPlayerInfo(Player ply) {
		this.register(ply);
		return this.info.get(ply);
	}
	
	public int getValue(Player ply, String key) {
		this.register(ply);
		if (!this.info.get(ply).containsKey(key)) {
			if (!hasString(this.values, key)) {return 0;}
			HashMap<String, Integer> toAdd = new HashMap<String, Integer>();
			toAdd.put(key, 0);
			this.info.put(ply, toAdd);
			return 0;
		}
		return this.info.get(ply).get(key);
	}
	
	public boolean save_all() {
		for (Player ply : this.info.keySet()) {
			if (!this.save(ply)) {
				return false;
			}
		}
		return true;
	}
	
	public boolean save(Player ply) {
		if (!this.save_folder.exists()) {
			this.save_folder.mkdir();
		}
		File plyfile = new File(this.save_folder.getPath() + "/" + ply.getName() + ".pinfo");
		if (plyfile.exists()) {plyfile.delete();}
		try {
			plyfile.createNewFile();
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(plyfile.getPath(), true)));
			for (String key : this.info.get(ply).keySet()) {
				out.println(key + ":" + this.info.get(ply).get(key));
			}
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		return true;
	}

	public boolean hasString(String[] list, String lookfor) {
		for (String s : list) {if (s.equals(lookfor)) {return true;}}
		return false;
	}
}
