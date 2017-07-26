
public class validator {
	boolean val_alphanum(String str,int mandatory_flag,int size)
	{
		if(str == null && mandatory_flag == 1)
		{
			return false;
		}
		else if(str == null){return true;}
		else if(str.length() > size)
		{
			return false;
		}
//		else if(!str.matches("[a-zA-Z0-9\\s\\,][-]+"))
//		{
//	
//			return false;
//		}
		else
		{
			return true;
		}
	}
	boolean val_number(String str,int size,int mandatory_flag)
	{
		if(str.equals(null) && mandatory_flag == 1)
		{
			return false;
		}
		else if(str.length() > size)
		{
			return false;
		}
		else if(!str.matches("[0-9]+"))
		{
			return false;
		}
		else
		{
			return true;
		}
		
	}
		
		boolean val_email(String str,int size,int mandatory_flag)
		{
			if(str.equals(null) && mandatory_flag == 1)
			{
				return false;
			}
			else if(str.length() > size)
			{
				return false;
			}
			else if(!str.matches("[a-zA-Z0-9@.]+"))
			{
				return false;
			}
			else
			{
				return true;
			}
			
			
		}
		boolean val_status(String str,int mandatory_flag)
		{
			if(str.equals(null) && mandatory_flag == 1)
			{
				return false;
			}
			else if(!(str.equals("N")||str.equals("M")||str.equals("D")||str.equals("A")||str.equals("R")))
			{
				return false;
			}
			else
			{
				return true;
			}
		}
		boolean val_flag(String str,int mandatory_flag)
		{
			if(str.equals(null) && mandatory_flag == 1)
			{
				return false;
			}
			else if(!(str.equals("A")||str.equals("I")))
			{
				return false;
			}
			else
			{
				return true;
			}
			
		}
		boolean val_date(String str,int size,int mandatory_flag)
		{
		
			if(str == null && mandatory_flag == 1)
			{
			
				return false;
				
			}
			else if(str == null){return true;}
			else if(str.length() > size)
			{
				
				return false;
			}
			else if(false)// write regex for date validation
			{
				return false;
			}
			return true;
	}

}
