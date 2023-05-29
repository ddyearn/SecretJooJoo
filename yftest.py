import numpy as np
import pandas as pd
from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib as mpl
import yfinance as yf

"""
tickers 라는 이름의 list에 yahoo finance에서 
사용하는 증권의 ticker를 입력한다
이 예제에서는 Visa의 V, Mastercard의 MA, 
MSCI inc의 MSCI를 사용한다
기존 tickers라는 이름의 list를 삭제한다
"""

tickers = ['V', 'MA', 'MSCI']

# 기존 pxclose라는 이름의 list를 삭제한다 
# pandas의 DataFrame이라는 기능은 matrix 형태의 자료를 처리하는 데에 용이하다. 
pxclose = pd.DataFrame()

"""
for... loop 구문으로 yahoo! finance의 자료를 순차적으로 다운로드 한다.
Adjusted Closing Price는 수정 종가를 의미한다
['Adj Close'] 코딩을 활용하여 yf.download가 가져온 자료 중 
adj close의 열(column) 값만 저장한다
"""

for t in tickers:
    pxclose[t] = yf.download(t, start = "2020-07-30", end="2021-07-30")['Adj Close']

"""
자료가 잘 받아졌는지 신속하게 파악하기 위해, 
pxclose matrix의 최근 자료들 
(여기에서는 2021년 7월 30일 이전 5영업일)을 열람한다.
"""
pxclose.tail()

# daily return 을 계산하자. pct_change() 함수를 사용한다
ret_daily = pxclose.pct_change()

"""
annual return을 계산한다. 1년 = 250영업일을 가정하여, 
daily return의 mean 값에 250을 곱해준다
"""
ret_annual = ret_daily.mean()*250

# daily return 값의 covariance를 계산한다.
cov_daily=ret_daily.cov()

"""
annual return 값의 covraiance를 계산한다. 
여기에서는 daily covariance * 250 = annual covariance로 가정한다
"""
cov_annual=cov_daily*250

# portfolio 수익률, 변동성, 투자 비중을 저장할 변수를 미리 저장한다. 
p_returns=[]
p_volatility=[]
p_weights = []

# len() 함수로 투자자산의 수를 계산한다. 
n_assets = len(tickers)

# n개 종목으로 투자 비중을 바꿔가며 3만 개의 포트폴리오를 만들 것이
n_ports = 30000

"""
# n_ports 만큼 반복하면서 자신의 투자 비중을 랜덤하게 만들고, 
포트폴리오의 기대수익률, 변동성을 계산한다
# 계산한 수익률, 변동성, 투자 비중은 앞서 미리 준비한 변수들인
p_returns, p_volatility, p_weights에 저장한다.
"""
for s in range(n_ports):
    
    #np.random.random() 함수로 난수 생성
    wgt=np.random.random(n_assets)
    
    #투자 비중 합계 100%를 위해 각 난수를 난수 합으로 나눈다
    wgt /= np.sum(wgt)
    
    """
    # portfolio 기대수익률 = 비중 * 각 종목별 기대수익률
    # dot product로 계산해 준다. 
    여기에서 wgt 는 (n,1) vector이며, ret_annual도 (n,1) vector
    """
    ret = np.dot(wgt, ret_annual)
    
    #portfolio의 변동성을 계산한다. (wgt x covariance matrix x wgt)^(1/2)
    vol = np.sqrt(np.dot(wgt.T, np.dot(cov_annual,wgt)))
    
    #계산한 수익률, 변동성, 비중을 추가한다. 
    p_returns.append(ret)
    p_volatility.append(vol)
    p_weights.append(wgt)


"""
#참고를 위해 세부 내용을 적어본다
# numpy의 random.random(n) 함수는 수학적 의미에서 [0,1) 
구간에 속하는 난수를 n개 만큼 출력한다
# 예를 들어, 
array([0.61240447, 0.12133946, 0.58294007, 0.33440155, 0.11916194])
"""
np.random.random(5)

"""
# wgt / = np.sum(wgt) 부분: 
# /= 왼쪽 변수에 오른쪽 값을 나누고 그 결과를 왼쪽 변수에 할당한다
# 즉, a/=b는 a= a/b를 의미함 
# a=8, b=4 일 경우,  a /= b 는 8/4 = 2 = a 를 의미
"""
a=8
b=4
a /= b
a

# portfolio volaility 에서 np.dot은 dot product를 의미 
# wgt.T 는 wgt의 transpose matrix를 의미 
# np.sqrt(a)는 a^(1/2)을 의미 

# np.array를 사용하여 matrix로 인식시킨다    
p_volatility = np.array(p_volatility)
p_returns = np.array(p_returns)    

"""
# 색상을 n_ports 만큼 만든다
# random.randint(0, 5)는 수학적 의미에서 [0,5) 구간에 
위치한 정수(integer) 중 하나를 random하게 추출한다
# random.rnadint(0,5,100) 에서 100은 100개를 추출하라는 뜻이다. 
"""
colors=np.random.randint(0, n_ports, n_ports)   
colors 

#n_ports 개수만큼 정수가 만들어졌는지 확인해보자
len(colors)
    
# Matplotlib에서 사용 가능한 스타일을 출력해 보자
print(plt.style.available)

# classic 스타일을 선택하자
plt.style.use('classic')

# 분산 차트를 설정하자 
# matplotlib 의 scatter 함수(x축, y축, 기타 조건) 형식으로 작성한다.
# c 조건은 color matrix의 값 순서대로 색깔을 대입한다
plt.scatter(p_volatility, p_returns, c=colors, marker='o')

"""
# x, y축 이름을 작성한다 (주의: xlabel, ylable, title, 그리고 show() 
까지 한꺼번에 선택하여 실행해야 1 plot에 그래프가 그려진다)
"""
plt.xlabel("portfolio's volatility (standard deviation): 2017.1.1 ~ 2021.7.28.")
plt.ylabel("portfolio's expected returns: 2017.1.1 ~ 2021.7.28.")

# 차트 제목을 지정한다
plt.title("Efficient Frontier for portfolio of Visa, Mastercard, and MSCI Inc.")

plt.show()