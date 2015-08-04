import java.util.ArrayList;
import java.util.Scanner;

public class MySQLConsole {
    public static void main(String[] args) {
        Scanner read = new Scanner(System.in);
        MySQL server = new MySQL("69.136.85.186", "3306", "remote_2", "tassword", "world");
        while (true) {
            String input = read.nextLine();
            ArrayList<ArrayList<String>> table = server.queryTable(input);
            for (ArrayList<String> row : table) {
                for (String item : row)
                    System.out.println(item + "\t + |");
            }
        }
    }
}
