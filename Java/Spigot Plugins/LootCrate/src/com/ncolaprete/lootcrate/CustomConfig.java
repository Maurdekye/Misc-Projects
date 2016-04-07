package com.ncolaprete.lootcrate;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Level;

import org.bukkit.configuration.file.FileConfiguration;
import org.bukkit.configuration.file.YamlConfiguration;
import org.bukkit.plugin.Plugin;

public class CustomConfig {
    Plugin plug;
    String name;
    FileConfiguration config;
    File configFile;
    
    CustomConfig(Plugin plug, String name) {
        this.plug = plug;
        this.name = name;
        this.config = null;
        this.configFile = null;
    }
    
    public void reloadConfig() {
        if (this.configFile == null) {
            this.configFile = new File(plug.getDataFolder(), this.name);
           }
        this.config = YamlConfiguration.loadConfiguration(this.configFile);
        InputStream defConfigStream = plug.getResource(this.name);
        if (defConfigStream != null) {
            YamlConfiguration defConfig = YamlConfiguration.loadConfiguration(defConfigStream);
            this.config.setDefaults(defConfig);
        }
    }
    
    public FileConfiguration getConfig() {
        if (this.config == null) {
            this.reloadConfig();
        }
        return this.config;
    }
    
    public void saveConfig() {
        if (this.config == null || this.configFile == null) {
            return;
        }
        try {
            getConfig().save(this.configFile);
        } catch (IOException ex) {
            plug.getLogger().log(Level.SEVERE, "Could not save config to " + this.configFile, ex);
        }
    }
    
    public void saveDefaultConfig() {
        if (this.configFile == null) {
            this.configFile = new File(plug.getDataFolder(), this.name);
        }
        if (!this.configFile.exists()) {
             plug.saveResource(this.name, false);
         }
    }
}
