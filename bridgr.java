import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.sql.Statement;
import java.sql.ResultSet;
public class jdbcbridge {

	     void loaddriver() {
	        try {
	            // The newInstance() call is a work around for some
	            // broken Java implementations

	            Class.forName("com.mysql.jdbc.Driver").newInstance();
	        } catch (Exception ex) {
	            // handle the error
	        }
	    }
	
	Connection conn = null;
	void connect()
	{
		try {
		    conn =
		       DriverManager.getConnection("jdbc:mysql://localhost/nucleus?" +
		                                   "user=fury&password=ffff");
		    PreparedStatement pstm = conn.prepareStatement("select * from test");
		    
		    
		} catch (SQLException ex) {
		    // handle any errors
		    System.out.println("SQLException: " + ex.getMessage());
		    System.out.println("SQLState: " + ex.getSQLState());
		    System.out.println("VendorError: " + ex.getErrorCode());
		}
	}
	
	
	
}
