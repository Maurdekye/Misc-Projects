package mainpackage;

import java.util.ArrayList;
import java.util.HashSet;

import org.bukkit.ChatColor;
import org.bukkit.Location;
import org.bukkit.OfflinePlayer;
import org.bukkit.entity.Player;
import org.bukkit.scoreboard.DisplaySlot;
import org.bukkit.scoreboard.Objective;
import org.bukkit.scoreboard.Score;
import org.bukkit.scoreboard.Scoreboard;
import org.bukkit.scoreboard.Team;

public class GameManager {
	public VSFortress plug;
	public Scoreboard board;
	public Objective teamscores;
	public int scoreLimit;
	public ArrayList<DoubleTeam> teams;
	private int maxTeams;
	public ChatColor[] usableColours;
	public Objective teamcount;
	public HashSet<String> listing;

	GameManager(VSFortress plug) {
		/*this.usableColours = new ChatColor[] { ChatColor.RED, ChatColor.BLUE,
				ChatColor.LIGHT_PURPLE, ChatColor.GREEN, ChatColor.YELLOW,
				ChatColor.AQUA, ChatColor.DARK_RED, ChatColor.DARK_BLUE,
				ChatColor.DARK_GREEN, ChatColor.GOLD, ChatColor.DARK_AQUA,
				ChatColor.DARK_PURPLE };*/
		this.usableColours = new ChatColor[] { ChatColor.RED, ChatColor.BLUE };
		this.plug = plug;
		this.scoreLimit = plug.getConfig().getInt("score_limit");
		this.listing = new HashSet<String>();
		this.maxTeams = Math.max(
				Math.min(usableColours.length,
						plug.getConfig().getInt("max_teams")), 2);
		this.board = empty();
		this.teamscores = this.board.registerNewObjective("Scores", "score");
		this.teamcount = this.board.registerNewObjective("Players", "count");
		this.setDisplay(Stat.SCORE);
		this.teams = new ArrayList<DoubleTeam>();
	}

	public DoubleTeam addPlayer(Player p) {
		DoubleTeam choose = null;
		boolean hasteam = false;
		if (this.listing.contains(p.getName()))
			return getTeam(p);
		for (DoubleTeam t : this.teams) {
			if (t.size() == 0 || t.members.contains(p.getName())) {
				choose = t;
				hasteam = true;
				break;
			}
		}
		if (teams.size() < this.maxTeams && !hasteam) {
			ChatColor colour = null;
			while (colour == null || hasColourRegistered(colour))
				colour = usableColours[(int) (Math.random() * usableColours.length)];
			choose = new DoubleTeam(colour, this);
			teams.add(choose);
		} else {
			for (DoubleTeam t : teams) {
				if (choose == null || t.size() < choose.size()) {
					choose = t;
					break;
				}
			}
		}
		p.sendMessage("Joined " + choose + ".");
		Location spawn = plug.getLocate(choose.colour.name().toLowerCase()
				+ "_team_spawn");
		choose.addPlayer(p);
		this.listing.add(p.getName());
		p.teleport(spawn);
		return choose;
	}

	public DoubleTeam decommit(Player p) {
		for (DoubleTeam t : teams) {
			if (t.members.contains(p.getName())) {
				t.decommit(p);
				return t;
			}
		}
		return null;
	}

	public DoubleTeam removePlayer(Player p) {
		Location spawn = plug.getLocate("lobby_spawn");
		for (DoubleTeam t : teams) {
			if (t.members.contains(p.getName())) {
				p.teleport(spawn);
				t.removePlayer(p);
				p.sendMessage("You left the game.");
				listing.remove(p.getName());
				return t;
			}
		}
		p.sendMessage("Not in a team");
		return null;
	}

	public void destroy() {
		for (DoubleTeam t : this.teams)
			t.destroy();
		this.teams = new ArrayList<DoubleTeam>();
		for (Team t : this.board.getTeams())
			t.unregister();
		this.board = empty();
	}

	public Scoreboard empty() {
		return this.plug.getServer().getScoreboardManager().getNewScoreboard();
	}

	public void setDisplay(Stat s) {
		this.board.clearSlot(DisplaySlot.SIDEBAR);
		switch (s) {
		case PLAYER_COUNT:
			this.teamcount.setDisplaySlot(DisplaySlot.SIDEBAR);
			break;
		case SCORE:
			this.teamscores.setDisplaySlot(DisplaySlot.SIDEBAR);
			break;
		}
	}

	public DoubleTeam getTeam(Player p) {
		for (DoubleTeam t : teams) {
			if (t.playerAdd.hasPlayer(p))
				return t;
		}
		return null;
	}

	public DoubleTeam getTeam(ChatColor c) {
		for (DoubleTeam t : teams) {
			if (t.colour == c)
				return t;
		}
		return null;
	}

	public void scoreTeam(DoubleTeam toScore, int points) {
		for (DoubleTeam t : teams) {
			if (t.colour == toScore.colour) {
				t.addScore(points);
				return;
			}
		}
	}

	public void resetScores() {
		for (DoubleTeam t : teams) {
			t.setScore(0);
		}
	}

	public void scoreTeam(Player p, int points) {
		scoreTeam(getTeam(p), points);
	}

	public void scoreTeam(ChatColor c, int points) {
		scoreTeam(getTeam(c), points);
	}

	private boolean hasColourRegistered(ChatColor colour) {
		for (DoubleTeam dt : this.teams) {
			if (dt.colour == colour)
				return true;
		}
		return false;
	}
}

enum Stat {
	PLAYER_COUNT, SCORE
}

class DoubleTeam {
	Team displayBar;
	Team playerAdd;
	OfflinePlayer representative;
	GameManager hostgame;
	ChatColor colour;
	String name;
	HashSet<String> members;

	DoubleTeam(ChatColor colour, GameManager hostgame) {
		this.hostgame = hostgame;
		this.colour = colour;
		this.members = new HashSet<String>();
		this.displayBar = hostgame.board.registerNewTeam(colour.name()
				+ "_fake");
		this.displayBar.setDisplayName(niceName(colour) + " Team");
		this.displayBar.setPrefix(colour + "");
		this.name = this.displayBar.getPrefix()
				+ this.displayBar.getDisplayName();
		this.representative = hostgame.plug.getServer().getOfflinePlayer(
				this.displayBar.getDisplayName());
		this.displayBar.addPlayer(this.representative);
		this.playerAdd = hostgame.board
				.registerNewTeam(colour.name() + "_real");
		this.playerAdd.setPrefix(colour + "");
		this.playerAdd.setAllowFriendlyFire(false);
		hostgame.teamscores.getScore(this.representative).setScore(0);
		hostgame.teamcount.getScore(this.representative).setScore(0);
	}

	public void addPlayer(Player p) {
		if (members.contains(p.getName()))
			return;
		p.setScoreboard(hostgame.board);
		this.playerAdd.addPlayer(p);
		this.members.add(p.getName());
		Score count = this.hostgame.teamcount.getScore(this.representative);
		count.setScore(count.getScore() + 1);
		for (String name : this.members) {
			if (name.equalsIgnoreCase(p.getName()))
				continue;
			Player ply = hostgame.plug.getServer().getPlayer(name);
			ply.sendMessage(p.getName() + " has joined your team!");
		}
	}

	public void setScore(int score) {
		hostgame.teamscores.getScore(this.representative).setScore(score);
	}

	public int getScore() {
		return hostgame.teamscores.getScore(this.representative).getScore();
	}

	public int size() {
		return this.members.size();
	}

	public void addScore(int amount) {
		this.setScore(this.getScore() + amount);
	}

	public void decommit(Player p) {
		Score count = this.hostgame.teamcount.getScore(this.representative);
		count.setScore(count.getScore() - 1);
	}

	public void removePlayer(Player p) {
		decommit(p);
		this.members.remove(p.getName());
		this.playerAdd.removePlayer(p);
		p.setScoreboard(hostgame.empty());
	}

	public void destroy() {
		Location spawn = hostgame.plug.getLocate("lobby_spawn");
		for (String name : this.members) {
			Player ply = this.hostgame.plug.getServer().getPlayer(name);
			this.playerAdd.removePlayer(ply);
			ply.setScoreboard(hostgame.empty());
			ply.teleport(spawn);
		}
		this.members = new HashSet<String>();
		this.displayBar.removePlayer(this.representative);
		this.hostgame.board.resetScores(this.representative);
		this.displayBar.unregister();
		this.playerAdd.unregister();
	}

	public String toString() {
		return this.colour + niceName(this.colour) + " Team" + ChatColor.RESET;
	}

	private String niceName(ChatColor colour) {
		String fin = "";
		for (String word : colour.name().split("_")) {
			fin += word.substring(0, 1).toUpperCase()
					+ word.substring(1).toLowerCase() + " ";
		}
		return fin.trim();
	}
}