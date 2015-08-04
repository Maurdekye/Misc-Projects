package mainpackage;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import org.bukkit.Bukkit;
import org.bukkit.ChatColor;
import org.bukkit.Material;
import org.bukkit.OfflinePlayer;
import org.bukkit.command.Command;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.entity.EntityDeathEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.inventory.ItemStack;
import org.bukkit.plugin.Plugin;
import org.bukkit.plugin.java.JavaPlugin;

public class MySQLManager extends JavaPlugin implements Listener {

	private static Connection sql;
	public Plugin plug = this;

	public static String ip = "127.0.0.1";
	public static String port = "3306";
	public static String username = "server";
	public static String password = "password";
	public static String database = "minecraft";

	public boolean onCommand(CommandSender sender, Command cmd, String command,
			String[] args) {
		boolean isCmd = !(sender instanceof Player);
		Player ply = null;
		String ply_name = null;
		ItemStack hand = null;
		if (!isCmd) {
			ply = (Player) sender;
			ply_name = ply.getName();
			hand = ply.getItemInHand();
		}

		if (command.equalsIgnoreCase("money")) {
			if (args.length == 0) {
				if (isCmd) {
					return false;
				}
				
				if (!ply.hasPermission("mysqlmanager.money")) {
					sender.sendMessage(ChatColor.RED
							+ "You don't have permission to see your account.");
				}
				
				ply.sendMessage(ChatColor.GOLD + "You have " + ChatColor.GREEN
						+ moneyString(getMoney(ply_name)) + ChatColor.GOLD
						+ ".");
			} else {
				if (!ply.hasPermission("mysqlmanager.other_money")) {
					sender.sendMessage(ChatColor.RED
							+ "You can't look at other people's balances!");
				}

				String target = getName(args[0]);
				if (target == null) {
					sender.sendMessage(ChatColor.RED + "Player '" + args[0]
							+ "' is not on the server.");
					return false;
				}

				sender.sendMessage(ChatColor.GOLD + target + " has "
						+ ChatColor.GREEN + moneyString(getMoney(target))
						+ ChatColor.GOLD + ".");
			}
		}

		if (command.equalsIgnoreCase("givemoney")) {
			if (isCmd) {
				sender.sendMessage(ChatColor.RED + "Only players have money.");
				return false;
			}
			
			if (!ply.hasPermission("mysqlmanager.givemoney")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to pay other players.");
			}
			
			if (args.length <= 1) {
				sender.sendMessage(ChatColor.RED
						+ "Usage: /givemoney <player> <amount>");
				return false;
			}

			String target = getName(args[0]);
			if (target == null) {
				sender.sendMessage(ChatColor.RED + "Player '" + args[0]
						+ "' is not on the server.");
				return false;
			}

			double amount = 0;
			try {
				amount = Double.parseDouble(args[1]);
			} catch (NumberFormatException e) {
				sender.sendMessage(ChatColor.RED
						+ "2nd argument must be a number!");
				return false;
			}

			if (amount < 0) {
				sender.sendMessage(ChatColor.RED
						+ "You can't steal money from another player!");
				return false;
			}

			if (amount > getMoney(ply.getName())) {
				sender.sendMessage(ChatColor.RED
						+ "You have insufficient funds to give to this person.");
				return false;
			}

			addMoney(ply.getName(), -amount);
			addMoney(target, amount);

			sender.sendMessage(ChatColor.GOLD + "You have given "
					+ ChatColor.GREEN + moneyString(amount) + ChatColor.GOLD
					+ " to " + target + ". Your balance is now "
					+ ChatColor.GREEN
					+ moneyString(getMoney(ply.getName())) + ChatColor.GOLD
					+ ".");
			Player targetPlayer = getServer().getPlayer(args[0]);
			if (!(targetPlayer == null)) {
				targetPlayer.sendMessage(ChatColor.GOLD + ply.getName()
						+ " has given you " + ChatColor.GREEN
						+ moneyString(amount) + ChatColor.GOLD
						+ ". Your balance is now " + ChatColor.GREEN
						+ moneyString(getMoney(target)) + ChatColor.GOLD + ".");
			}

		}

		if (command.equalsIgnoreCase("price")) {
			if (isCmd) {
				sender.sendMessage(ChatColor.RED
						+ "You have to be a player to look at the pricing of items.");
				return false;
			}

			if (!ply.hasPermission("mysqlmanager.price")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to see the prices of items.");
			}
			
			if (hand.getType() == Material.AIR) {
				sender.sendMessage(ChatColor.RED
						+ "You must hold an item in your hand first!");
				return false;
			}

			String nicename = niceName(hand.getType());
			double buyprice = getBuy(hand.getType());
			double sellprice = getSell(hand.getType());

			String buytext = "'t be bought";
			String selltext = "'t be sold";
			if (buyprice > 0)
				buytext = " be bought for " + ChatColor.GREEN
						+ moneyString(buyprice) + ChatColor.GOLD + " each";
			if (sellprice > 0)
				selltext = " be sold for " + ChatColor.GREEN
						+ moneyString(sellprice) + ChatColor.GOLD + " each";

			if (buyprice == sellprice) {
				if (buyprice <= 0) {
					ply.sendMessage(ChatColor.GOLD + capitalize(nicename)
							+ " can't be bought or sold.");
				} else {
					ply.sendMessage(ChatColor.GOLD + capitalize(nicename)
							+ " can be bought and sold for " + ChatColor.GREEN
							+ moneyString(buyprice) + ChatColor.GOLD + ".");
				}
			} else {
				ply.sendMessage(ChatColor.GOLD + capitalize(nicename) + " can"
						+ buytext + ", and can" + selltext + ".");
			}
			if (hand.getAmount() > 1 && sellprice > 0)
				ply.sendMessage(ChatColor.GOLD
						+ "Selling the whole stack would net you "
						+ ChatColor.GREEN
						+ moneyString(buyprice * hand.getAmount())
						+ ChatColor.GOLD + ".");
		}

		if (command.equalsIgnoreCase("sell")) {
			if (isCmd) {
				sender.sendMessage(ChatColor.RED
						+ "You have to be a player to sell items.");
				return false;
			}

			if (!ply.hasPermission("mysqlmanager.sell")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to sell items.");
			}

			if (hand.getType() == Material.AIR) {
				sender.sendMessage(ChatColor.RED
						+ "You must hold an item in your hand first!");
				return false;
			}

			int amount = hand.getAmount();
			if (args.length > 0) {
				try {
					amount = Integer.parseInt(args[0]);
				} catch (NumberFormatException e) {
					sender.sendMessage(ChatColor.RED
							+ "Argument must be a number!");
					return false;
				}
			}

			String nicename = niceName(hand.getType());
			double price = getSell(hand.getType());

			if (price <= 0) {
				sender.sendMessage(ChatColor.RED + capitalize(nicename)
						+ "s can't be sold!");
				return false;
			}

			int sold = useItem(ply, hand.getType(), amount);

			addMoney(ply.getName(), price * sold);

			if (sold == 1) {
				ply.sendMessage(ChatColor.GOLD + "You sold 1 " + nicename
						+ " for a total of " + ChatColor.GREEN
						+ moneyString(price) + ChatColor.GOLD + ".");
			} else {
				ply.sendMessage(ChatColor.GOLD + "You sold " + sold + " "
						+ nicename + "s for a total of " + ChatColor.GREEN
						+ moneyString(price * sold) + ChatColor.GOLD + ", at "
						+ ChatColor.GREEN + moneyString(price) + ChatColor.GOLD
						+ " per.");
			}
		}

		if (command.equalsIgnoreCase("buy")) {
			if (isCmd) {
				sender.sendMessage(ChatColor.RED
						+ "You have to be a player to buy items.");
				return false;
			}

			if (!ply.hasPermission("mysqlmanager.buy")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to buy items.");
			}

			if (hand.getType() == Material.AIR) {
				sender.sendMessage(ChatColor.RED
						+ "You must hold an item in your hand first!");
				return false;
			}

			String nicename = niceName(hand.getType());
			double price = getBuy(hand.getType());

			if (price <= 0) {
				sender.sendMessage(ChatColor.RED + capitalize(nicename)
						+ "s can't be bought!");
				return false;
			}

			int amount = Math.max(1,
					hand.getType().getMaxStackSize() - hand.getAmount());
			if (args.length > 0) {
				try {
					amount = Integer.parseInt(args[0]);
				} catch (NumberFormatException e) {
					sender.sendMessage(ChatColor.RED
							+ "Argument must be a number!");
					return false;
				}
			}

			if (amount == 0) {
				sender.sendMessage(ChatColor.RED
						+ "You can't just buy zero of something.");
				return false;
			}

			double money = getMoney(ply.getName());

			if (price > money) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have enough money to buy any " + nicename
						+ "!");
				return false;
			}

			if (price * amount > money) {
				int buyable = (int) (money / price);
				sender.sendMessage(ChatColor.RED
						+ "You don't have enough money to buy that much "
						+ nicename + ", but you can afford to buy " + buyable
						+ " of them.");
				return false;
			}

			addMoney(ply.getName(), -(amount * price));

			sender.sendMessage(ChatColor.GOLD + "You have bought " + amount
					+ " " + nicename + "s, for a total of " + ChatColor.GREEN
					+ moneyString(price * amount) + ChatColor.GOLD + ".");

			while (amount > 0) {
				ItemStack bought = new ItemStack(hand.getType(), Math.min(
						amount, hand.getType().getMaxStackSize()));
				bought.setData(hand.getData());
				ply.getWorld().dropItem(ply.getEyeLocation(), bought);
				amount -= hand.getType().getMaxStackSize();
			}

		}

		if (command.equalsIgnoreCase("stats")) {
			if (args.length == 0) {
				if (isCmd) {
					sender.sendMessage(ChatColor.RED
							+ "Only players have this information.");
					return false;
				}

				if (!ply.hasPermission("mysqlmanager.stats")) {
					sender.sendMessage(ChatColor.RED
							+ "You don't have permission to see statistics about yourself.");
				}
				
				int mobkills = 0;
				int playerkills = 0;
				int deaths = 0;
				try {
					PreparedStatement query = sql
							.prepareStatement("SELECT `mobkills`, `playerkills`, `deaths` FROM `stats` WHERE player = ?");
					query.setString(1, ply.getName());
					ResultSet results = query.executeQuery();
					if (results.next()) {
						mobkills = results.getInt(1);
						playerkills = results.getInt(2);
						deaths = results.getInt(3);
					}
					query.close();
					results.close();
				} catch (Exception e) {
					e.printStackTrace();
				}
				String printmk = ChatColor.GREEN + "" + mobkills + ChatColor.GOLD;
				String printpk = ChatColor.GREEN + "" + playerkills
						+ ChatColor.GOLD;
				String printd = ChatColor.GREEN + "" + deaths + ChatColor.GOLD;
				sender.sendMessage(ChatColor.GOLD + "You have killed " + printmk
						+ " mobs, " + printpk + " players, and died " + printd
						+ " times.");
			} else {
				if (!ply.hasPermission("mysqlmanager.other_stats")) {
					sender.sendMessage(ChatColor.RED
						+ "You don't have permission to see statistics about other players.");
				}
				
				String target = getName(args[0]);
				if (target == null) {
					sender.sendMessage(ChatColor.RED + "Player '" + args[0]
							+ "' has never been on this server.");
					return false;
				}
				
				int mobkills = 0;
				int playerkills = 0;
				int deaths = 0;
				try {
					PreparedStatement query = sql
							.prepareStatement("SELECT `mobkills`, `playerkills`, `deaths` FROM `stats` WHERE player = ?");
					query.setString(1, ply.getName());
					ResultSet results = query.executeQuery();
					if (results.next()) {
						mobkills = results.getInt(1);
						playerkills = results.getInt(2);
						deaths = results.getInt(3);
					}
					query.close();
					results.close();
				} catch (Exception e) {
					e.printStackTrace();
				}
				String printmk = ChatColor.GREEN + "" + mobkills + ChatColor.GOLD;
				String printpk = ChatColor.GREEN + "" + playerkills
						+ ChatColor.GOLD;
				String printd = ChatColor.GREEN + "" + deaths + ChatColor.GOLD;
				sender.sendMessage(ChatColor.GOLD + "You have killed " + printmk
						+ " mobs, " + printpk + " players, and died " + printd
						+ " times.");
			}
		}

		// Admin Commands

		if (command.equalsIgnoreCase("setprice")) {

			if (!ply.hasPermission("mysqlmanager.setprice")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to set the price of items.");
			}

			if (args.length < 1) {
				sender.sendMessage(ChatColor.RED + "Usage: /setprice <price>");
				return false;
			}

			if (hand.getType() == Material.AIR) {
				sender.sendMessage(ChatColor.RED
						+ "You must hold an item in your hand first!");
				return false;
			}

			Material change = hand.getType();

			double price = 0;
			try {
				price = Double.parseDouble(args[0]);
			} catch (NumberFormatException e) {
				sender.sendMessage(ChatColor.RED + "Argument must be a number!");
				return false;
			}

			sqlUpdate("UPDATE `items` SET buy = " + price + " WHERE name = '"
					+ change.toString() + "'");
			sqlUpdate("UPDATE `items` SET sell = " + price + " WHERE name = '"
					+ change.toString() + "'");
			sender.sendMessage(ChatColor.GOLD
					+ "Changed buying and selling price of " + niceName(change)
					+ " to " + ChatColor.GREEN + moneyString(price)
					+ ChatColor.GOLD + ".");
		}

		if (command.equalsIgnoreCase("setbuy")) {
			if (!ply.hasPermission("mysqlmanager.setbuy")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to set the buying price of items.");
			}
			
			if (args.length < 1) {
				sender.sendMessage(ChatColor.RED + "Usage: /setprice <price>");
				return false;
			}

			if (hand.getType() == Material.AIR) {
				sender.sendMessage(ChatColor.RED
						+ "You must hold an item in your hand first!");
				return false;
			}

			Material change = hand.getType();

			double price = 0;
			try {
				price = Double.parseDouble(args[0]);
			} catch (NumberFormatException e) {
				sender.sendMessage(ChatColor.RED + "Argument must be a number!");
				return false;
			}

			sqlUpdate("UPDATE `items` SET buy = " + price + " WHERE name = '"
					+ change.toString() + "'");
			sender.sendMessage(ChatColor.GOLD + "Changed buying price of "
					+ niceName(change) + " to " + ChatColor.GREEN
					+ moneyString(price) + ChatColor.GOLD + ".");
		}

		if (command.equalsIgnoreCase("setsell")) {
			if (!ply.hasPermission("mysqlmanager.setsell")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to set the selling price of items.");
			}
			
			if (args.length < 1) {
				sender.sendMessage(ChatColor.RED + "Usage: /setprice <price>");
				return false;
			}

			if (hand.getType() == Material.AIR) {
				sender.sendMessage(ChatColor.RED
						+ "You must hold an item in your hand first!");
				return false;
			}

			Material change = hand.getType();

			double price = 0;
			try {
				price = Double.parseDouble(args[0]);
			} catch (NumberFormatException e) {
				sender.sendMessage(ChatColor.RED + "Argument must be a number!");
				return false;
			}

			sqlUpdate("UPDATE `items` SET sell = " + price + " WHERE name = '"
					+ change.toString() + "'");
			sender.sendMessage(ChatColor.GOLD + "Changed selling price of "
					+ niceName(change) + " to " + ChatColor.GREEN
					+ moneyString(price) + ChatColor.GOLD + ".");
		}

		if (command.equalsIgnoreCase("setmoney")) {
			if (!ply.hasPermission("mysqlmanager.setmoney")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to set your or someone else's accout balance.");
			}
			
			double amount = 0;
			if (args.length == 1) {
				if (isCmd) {
					sender.sendMessage(ChatColor.RED
							+ "Only players have money.");
					return false;
				}
				try {
					amount = Double.parseDouble(args[0]);
				} catch (NumberFormatException e) {
					sender.sendMessage(ChatColor.RED
							+ "Argument must be a number.");
					return false;
				}
				setMoney(ply.getName(), amount);
				ply.sendMessage(ChatColor.GOLD + "Your balance is now "
						+ ChatColor.GREEN
						+ moneyString(getMoney(ply.getName())) + ChatColor.GOLD
						+ ".");
			} else if (args.length >= 2) {
				String target = getName(args[0]);
				if (target == null) {
					sender.sendMessage(ChatColor.RED + "Player '" + args[0]
							+ "' has never been on this server.");
					return false;
				}
				try {
					amount = Double.parseDouble(args[1]);
				} catch (NumberFormatException e) {
					sender.sendMessage(ChatColor.RED
							+ "2nd argument must be a number.");
					return false;
				}
				setMoney(target, amount);
				sender.sendMessage(ChatColor.GOLD + target
						+ "'s balance is now " + ChatColor.GREEN
						+ moneyString(getMoney(target)) + ChatColor.GOLD + ".");
				Player targetPlayer = getServer().getPlayer(args[0]);
				if (!(targetPlayer == null)) {
					targetPlayer.sendMessage(ChatColor.GOLD
							+ "Your balance has been altered. It is now "
							+ ChatColor.GREEN + moneyString(getMoney(target))
							+ ChatColor.GOLD + ".");
				}
			} else {
				sender.sendMessage(ChatColor.RED
						+ "Usage: /setmoney [player] <amount>");
			}
		}

		if (command.equalsIgnoreCase("addmoney")) {
			if (!ply.hasPermission("mysqlmanager.buy")) {
				sender.sendMessage(ChatColor.RED
						+ "You don't have permission to add money to you or other people's accounts.");
			}
			double amount = 0;
			if (args.length == 1) {
				if (isCmd) {
					sender.sendMessage(ChatColor.RED
							+ "Only players have money.");
					return false;
				}
				try {
					amount = Double.parseDouble(args[0]);
				} catch (NumberFormatException e) {
					sender.sendMessage(ChatColor.RED
							+ "Argument must be a number.");
					return false;
				}
				addMoney(ply_name, amount);
				ply.sendMessage(ChatColor.GOLD + "Your balance is now "
						+ ChatColor.GREEN + moneyString(getMoney(ply_name))
						+ ChatColor.GOLD + ".");
			} else if (args.length >= 2) {
				String target = getName(args[0]);
				if (target.equals("")) {
					sender.sendMessage(ChatColor.RED + "Player '" + args[0]
							+ "' has never been on this server.");
					return false;
				}

				try {
					amount = Double.parseDouble(args[1]);
				} catch (NumberFormatException e) {
					sender.sendMessage(ChatColor.RED
							+ "2nd argument must be a number.");
					return false;
				}
				addMoney(target, amount);
				sender.sendMessage(ChatColor.GOLD + target
						+ "'s balance is now " + ChatColor.GREEN
						+ getMoney(target) + ChatColor.GOLD + ".");
				Player targetPlayer = getServer().getPlayer(args[0]);
				if (!(targetPlayer == null)) {
					targetPlayer.sendMessage(ChatColor.GOLD
							+ "Your balance has been altered. It is now "
							+ ChatColor.GREEN + moneyString(getMoney(target))
							+ ChatColor.GOLD + ".");
				}
			} else {
				sender.sendMessage(ChatColor.RED
						+ "Usage: /setmoney [player] <amount>");
			}
		}
		return true;
	}

	public void onDisable() {
		try {
			if (sql != null && !sql.isClosed())
				sql.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
	}

	@SuppressWarnings("deprecation")
	public void onEnable() {

		getServer().getPluginManager().registerEvents(this, this);
		getConfig().options().copyDefaults(true);
		saveConfig();

		ip = getConfig().getString("ip");
		port = getConfig().getString("port");
		username = getConfig().getString("username");
		password = getConfig().getString("password");
		database = getConfig().getString("database");

		try {
			sql = DriverManager.getConnection("jdbc:mysql://" + ip + ":" + port
					+ "/" + database, username, password);
			System.out
					.println("Connected successfully to MySQL server at :ip:::port:."
							.replaceAll(":port:", port).replaceAll(":ip:", ip));
		} catch (Exception e) {
			System.out.println("Enountered a " + e.getCause().toString());
			System.out
					.print("\n\nCould not connect to MySQL database at :ip:::port:. Please check that:"
							.replaceAll(":ip:", ip).replaceAll(":port:", port)
							+ "\n\t- The ip / port in the config file is correct for your server,"
							+ "\n\t- Your user provided in the config has read/write permission \n\t   for that server, and "
							+ "\n\t- You have a database within that server named ':database:'.\n"
									.replaceAll(":database:", database));
			getServer().getPluginManager().disablePlugin(this);
		}

		sqlUpdate("CREATE TABLE IF NOT EXISTS `stats` ("
				+ "`player` varchar(17) NOT NULL, "
				+ "`money` double(16,2) NOT NULL, "
				+ "`deaths` int(16) unsigned NOT NULL, "
				+ "`playerkills` int(16) unsigned NOT NULL, "
				+ "`mobkills` int(10) unsigned NOT NULL, "
				+ "PRIMARY KEY (`player`)"
				+ ") ENGINE=InnoDB DEFAULT CHARSET=latin1;");

		sqlUpdate("CREATE TABLE IF NOT EXISTS `items` ("
                + "`id` smallint(5), "
				+ "`name` varchar(64), " + "`buy` double(16,2), "
				+ "`sell` double(16,2), " + "PRIMARY KEY (`id`)"
				+ ") ENGINE=InnoDB DEFAULT CHARSET=latin1;");

		for (Material mat : Material.values()) {
			sqlUpdate("INSERT IGNORE INTO `items` VALUES(id, 'itemname', 0.00, 0.00);"
					.replaceAll("id", mat.getId() + "").replaceAll("itemname",
							mat.toString()));
		}

		for (Player ply : Bukkit.getOnlinePlayers())
			addPlayer(ply.getName());
	}

	public synchronized static boolean addPlayer(String ply) {
		try {
			PreparedStatement query = sql
					.prepareStatement("SELECT `player` FROM `stats` WHERE player = ?");
			query.setString(1, ply);
			ResultSet results = query.executeQuery();
			boolean isIn = results.next();
			query.close();
			results.close();

			if (!isIn) {
				PreparedStatement newquery = sql
						.prepareStatement("INSERT INTO `stats` VALUES(?, 0, 0, 0, 0)");
				newquery.setString(1, ply);
				newquery.executeUpdate();
				newquery.close();
				return true;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return false;
	}

	public static String moneyString(double inval) {
		Double.toString(inval);
		String[] ends = (inval + "").split("\\.");
		String cents = ends[1];
		if (cents.length() == 1)
			cents += "0";
		String dollars = "";
		int e = 1;
		for (int i=ends[0].length();i>0;i--) {
			dollars = ends[0].substring(i-1, i) + dollars;
			if (e%3 == 0 && i != 1) dollars = "," + dollars;
			e++;
		}
		return "$" + dollars + "." + cents;
	}

	public static int useItem(Player ply, Material itemtoget, int needed) {
		int total = 0;
		for (ItemStack item : ply.getInventory().getContents()) {
			if (item == null)
				continue;
			if (item.getType() == itemtoget) {
				if (item.getAmount() > needed - total) {
					item.setAmount(item.getAmount() - (needed - total));
					total = needed;
					break;
				} else {
					total += item.getAmount();
					ply.getInventory().removeItem(
							new ItemStack(itemtoget, item.getAmount()));
				}
			}
		}
		;
		return total;
	}

	public static int howMuch(Player ply, Material check) {
		int total = 0;
		for (ItemStack item : ply.getInventory().getContents()) {
			if (item == null)
				continue;
			if (item.getType() != check)
				continue;
			total += item.getAmount();
		}
		return total;
	}

	public static boolean takeOnlyIf(Player ply, Material item, int needed) {
		if (howMuch(ply, item) >= needed) {
			useItem(ply, item, needed);
			return true;
		}
		return false;
	}

	public String capitalize(String instr) {
		return instr.substring(0, 1).toUpperCase()
				+ instr.substring(1, instr.length()).toLowerCase();
	}

    public String niceName(Material mat) {
        return "blah";
		/*item = CraftItemStack
				.asNMSCopy(new ItemStack(mat, 1));
		if (item == null)
			return mat.toString().toLowerCase().replace('_', ' ');
		else
			return item.getName().toLowerCase();*/
	}

	public synchronized String getName(String ply) {
		Player online = getServer().getPlayer(ply);
		if (online != null)
			return online.getName();
		OfflinePlayer offline = getServer().getOfflinePlayer(ply);
		if (offline != null)
			return offline.getName();
		boolean isOn = false;
		try {
			PreparedStatement query = sql
					.prepareStatement("SELECT player FROM stats WHERE player = ?");
			query.setString(1, ply);
			ResultSet results = query.executeQuery();
			isOn = results.next();
			results.close();
			query.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		if (isOn)
			return ply;
		return "";
	}

	public synchronized static double getBuy(Material mat) {
		double price = 0;
		try {
			PreparedStatement query = sql
					.prepareStatement("SELECT `buy` FROM items WHERE name = '"
							+ mat.toString() + "'");
			ResultSet results = query.executeQuery();
			if (results.next())
				price = results.getDouble(1);
			query.close();
			results.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return price;
	}

	public synchronized static double getSell(Material mat) {
		double price = 0;
		try {
			PreparedStatement query = sql
					.prepareStatement("SELECT `sell` FROM items WHERE name = '"
							+ mat.toString() + "'");
			ResultSet results = query.executeQuery();
			if (results.next())
				price = results.getDouble(1);
			query.close();
			results.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return price;
	}

	public synchronized static void setPrice(Material mat, double price) {
		sqlUpdate("UPDATE items SET price = " + price + " WHERE name = '"
				+ mat.toString() + "'");
	}

	public synchronized static double getMoney(String ply) {
		addPlayer(ply);
		double money = 0;
		try {
			PreparedStatement query = sql
					.prepareStatement("SELECT money FROM `stats` WHERE player = ?");
			query.setString(1, ply);
			ResultSet results = query.executeQuery();
			if (results.next())
				money = results.getDouble(1);
			else
				System.out.print("No entry.");
			query.close();
			results.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return money;
	}

	public synchronized static void sqlUpdate(String statement) {
		try {
			PreparedStatement query = sql.prepareStatement(statement);
			query.executeUpdate();
			query.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public synchronized static void setMoney(String ply, double amount) {
		addPlayer(ply);
		try {
			PreparedStatement query = sql
					.prepareStatement("UPDATE `stats` SET money = ? WHERE player = ?");
			query.setDouble(1, amount);
			query.setString(2, ply);
			query.executeUpdate();
			query.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public synchronized static void addMoney(String ply, double amount) {
		setMoney(ply, getMoney(ply) + amount);
	}

	public synchronized static int addToCount(String ply, String column) {
		addPlayer(ply);
		int amount = 0;
		try {
			PreparedStatement query = sql.prepareStatement("SELECT " + column
					+ " FROM `stats` WHERE player = ?");
			query.setString(1, ply);
			ResultSet results = query.executeQuery();
			if (results.next())
				amount = results.getInt(1);
			query.close();
			results.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		try {
			PreparedStatement query = sql
					.prepareStatement("UPDATE `stats` SET " + column
							+ " = ? WHERE player = ?");
			query.setInt(1, amount + 1);
			query.setString(2, ply);
			query.executeUpdate();
			query.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return amount + 1;
	}

	// Event Listeners

	@EventHandler
	public void Death(PlayerDeathEvent event) {
		addToCount(event.getEntity().getName(), "deaths");
		if (event.getEntity().getKiller() instanceof Player) {
			addToCount(event.getEntity().getKiller().getName(), "playerkills");
		}
	}

	@EventHandler
	public void mobKill(EntityDeathEvent event) {
		Player killer = event.getEntity().getKiller();
		if (killer != null) {
			addToCount(event.getEntity().getKiller().getName(), "mobkills");
		}
	}

	@EventHandler
	public void Join(PlayerJoinEvent event) {
		addPlayer(event.getPlayer().getName());
	}

}
