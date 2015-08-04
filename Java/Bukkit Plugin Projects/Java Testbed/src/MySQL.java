
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.HashMap;

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
	
	public String[] queryColumn(String query) {
		return queryColumn(query, 1);
	}
	public String[] queryColumn(String query, int columnnumber) {
		ArrayList<String> items = new ArrayList<String>();
		try {
			PreparedStatement statement = this.sql.prepareStatement(query);
			ResultSet results = statement.executeQuery();
			while (results.next())
				items.add(results.getString(columnnumber));
			results.close();
			statement.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return items.toArray(new String[items.size()]);
		
	}

	public String[] queryRow(String query) {
		return queryRow(query, 1);
	}
	public String[] queryRow(String query, int rownumber) {
		ArrayList<String> items = new ArrayList<String>();
		try {
			PreparedStatement statement = this.sql.prepareStatement(query);
			ResultSet results = statement.executeQuery();
			while (results.next())
				if (results.getRow() == rownumber) break;
			if (results.isLast()) return new String[0];
			for (int i=1; i<= results.getMetaData().getColumnCount(); i++)
				items.add(results.getString(i));
			results.close();
			statement.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return items.toArray(new String[items.size()]);
		
	}
	
	public ArrayList<ArrayList<String>> queryTable(String query) {
		ArrayList<ArrayList<String>> items = new ArrayList<ArrayList<String>>();
		try {
			PreparedStatement statement = this.sql.prepareStatement(query);
			ResultSet results = statement.executeQuery();
			while (results.next()) {
				ArrayList<String> row = new ArrayList<String>();
				for (int i=1; i<= results.getMetaData().getColumnCount(); i++)
					row.add(results.getString(i));
				items.add(row);
			}
			results.close();
			statement.close();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return items;
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
