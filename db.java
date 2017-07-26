import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;


public class preprocessor {
	private ArrayList<String> records = new ArrayList<String>();
	public ArrayList<String> input(String filename)
	{
		BufferedReader br = null;
		FileReader fr = null;
		try 
		{
			fr = new FileReader(filename);
			br = new BufferedReader(fr);
			String line;
			while ((line = br.readLine()) != null) {
				records.add(line);
			}
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
		finally
		{
			try
			{
				if(br!= null)
				{
					br.close();
				}
				if(fr != null)
				{
					fr.close();
				}
			}
			catch(IOException ex)
			{
				ex.printStackTrace();
			}
		}
		return records;
	}
	void split(ArrayList<String> records,char rejection_level,jdbcbridge push)
	{
		List<String> temp = new ArrayList<String>();
		List<Boolean> error = new ArrayList<Boolean>();
		validator v=new validator();//Create Validator object
		errorLogger e = new errorLogger();
		ArrayList<String> buffer  = new ArrayList<String>();
		Path path=Paths.get("error logs.txt");
		int flag = 0;
		int n_records=0;
		for(int i = 0;i < 1;i++)//<records.size()
		{
			for(String w:records.get(i).split("~",18))
				{	
					if(w.equals(""))
					{
						w=null;
					}
					temp.add(w);
				}
			error.add(v.val_alphanum(temp.get(0), 1, 10));//Implement hash set here 
			error.add(v.val_alphanum(temp.get(1), 1, 30));//for performance
			error.add(v.val_alphanum(temp.get(2), 1, 100));
			error.add(v.val_alphanum(temp.get(3), 0, 100));
			error.add(v.val_number(temp.get(4), 6, 1));
			error.add(v.val_email(temp.get(5), 100, 1));
			error.add(v.val_number(temp.get(6), 20, 0));
			error.add(v.val_alphanum(temp.get(7), 1, 100));
			error.add(v.val_status(temp.get(8), 1));
			error.add(v.val_flag(temp.get(9), 1));
			error.add(v.val_date(temp.get(10), 100, 1));
			error.add(v.val_alphanum(temp.get(11), 1, 30));
			error.add(v.val_date(temp.get(12), 100, 0));
			error.add(v.val_alphanum(temp.get(13), 0, 30));
			error.add(v.val_date(temp.get(14), 100, 0));
			error.add(v.val_alphanum(temp.get(15), 0, 30));
			if(error.contains(false))
			{	
				if(rejection_level == 'R')
				{
					System.out.println("Dumping record");
					e.recordDump(records.get(i),path);
				}	
				flag= flag +1;
			}
			else
			{
				if(rejection_level =='R')
				{
					push.connect();// insert row in db
				}
				else
				{
					String s1="";
					for(String s:temp)
						{	
							if(s == null)
								{
									s1=s1.concat("NULL\t");
								}
							else
								{
									s1 = s1.concat(s+"\t");
								}
						}
					buffer.add(s1);
				}
				n_records += 1;
				
			}	
		error.clear();	
		temp.clear();
		
		}
		System.out.println("Validation test sucessfully completed");
		System.out.println("SCANNED\t"+n_records+"\tRECORDS FOUND\t"+flag+"\tERROR/s");
		if(rejection_level == 'F' && flag > 0)
		{
			System.out.println("Dumping Entire file : More than 0 error Found");
			e.fileDump(records,path);
			return;
		}
		else
		{
			for (String s : buffer) //commit to db
			{
				s.split(" ");
			}
		}
		
	}
}
