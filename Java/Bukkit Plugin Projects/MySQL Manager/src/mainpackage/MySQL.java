package mainpackage;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

public class MySQL {
	private Connection sql;
	
	public MySQL (String ip, String port, String username, String password, String database) {
		try {
			this.sql = DriverManager.getConnection("jdbc:mysql://" + ip + ":" + port
					+ "/" + database, username, password);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public ResultSet query(String query) {
		try {
			PreparedStatement statement = this.sql.prepareStatement(query);
			ResultSet results = statement.executeQuery();
			statement.close();
			return results;
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public boolean update(String update) {
		try {
			PreparedStatement statement = this.sql.prepareStatement(update);
			statement.executeUpdate(update);
			return true;
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return false;
		
	}
	
	public boolean close() {
		if (this.sql == null) return true;
		try {
			if (this.sql.isClosed()) return true;
			this.sql.close();
			return true;
		} catch (SQLException e) {
			e.printStackTrace();
			return false;
		}
	}
}
