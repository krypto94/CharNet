import java.util.ArrayList;


public class run {

	public static void main(String[] args) {
		preprocessor a = new preprocessor();
		ArrayList<String> records = new ArrayList<String>();
		String filename = "C:\\Users\\Rishabh\\workspace\\BRD_DB\\src\\File2.txt";
		char rejection_level ='F';
		records=a.input(filename);
		
		jdbcbridge jdbc = new jdbcbridge();
		jdbc.loaddriver();
		jdbc.connect();
		a.split(records,rejection_level,jdbc);
		
	}

}
