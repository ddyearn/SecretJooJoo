import pandas as pd

#01) Excel 파일에서 원하는 부분만 불러온 후 Pandas DataFrame에 저장
data = pd.read_excel(io='./daily_price_data.xlsx', 
                   sheet_name='Sheet2',
                   usecols='A, B', 
                   index_col = 0,
                   skiprows=13)
print(data)

