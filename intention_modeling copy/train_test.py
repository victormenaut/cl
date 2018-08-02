def main():

	
	a = {'x': [1,2,3], 'y': [3,4,5]}
	b = {'x': [6,7,8], 'y': [9,0,1]}
	c = {'x': [], 'y':[]} 
	

	for i in a:
		merge = a[i] + b[i]
		c[i] = merge 
		
	print(c)




main()