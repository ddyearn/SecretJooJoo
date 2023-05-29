import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import yfinance as yf

#TIGER S&P500 Future ETF 143850.KS
#KODEX 200 069500.KS
#KOSEF 국고채 10년 148070.KS
#TIGER 단기선진하이일드 182490.KS
#KODEX gold future 132030.KS
#KODEX inverse 114800.KS
#US long - KOSEF US dollar Future 138230.KS
#US short - KOSEF US dollar inverse Future 139660.KS
#MMF - KOSEF 단기자금 130730.KS

tickers = ['069500.KS', '^N225','143850.KS', '148070.KS', '182490.KS', '132030.KS', '114800.KS', '130730.KS', '138230.KS','139660.KS']
pxclose = pd.DataFrame()

for t in tickers:
    pxclose[t] = yf.download(t, start = "2017-01-01", end="2023-05-15")['Adj Close']
pxclose.tail()
ret_daily = pxclose.pct_change()
ret_annual = ret_daily.mean()*250
cov_daily=ret_daily.cov()
cov_annual=cov_daily*250
p_returns=[]
p_volatility=[]
p_weights = []
n_assets = len(tickers)
n_ports = 30000
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
    
# #Plot minimum variance frontier (최소 분산 곡선 그래프)
# #Random portfolios (무작위 포트폴리오)
# means = port_ret_var['Port_ret'].values
# stds = port_ret_var['Port_std'].values
 
# #Minimum varaince portfolios (최소분산 포트폴리오)
# opt_returns = mean_stds_df['Port_ret']
# opt_stds = mean_stds_df['Port_std']
 
# fig = plt.figure(figsize=(15,8))
# plt.plot(stds, means, 'o')
# plt.ylabel('mean',fontsize=12)
# plt.xlabel('std',fontsize=12)
# plt.plot(opt_stds, opt_returns, 'y-o')
# plt.xlim(0.013,0.025)
# plt.ylim(0.001,0.0024)
# plt.title('Minimum variance frontier for risky assets',fontsize=15)


p_volatility = np.array(p_volatility)
p_returns = np.array(p_returns)    
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
plt.xlabel("portfolio's volatility (standard deviation): 2017.1.1 ~ 2023.5.15")
plt.ylabel("portfolio's expected returns: 2017.1.1 ~ 2023.5.15.")

# 차트 제목을 지정한다
plt.title("Efficient Frontier for portfolio")
plt.show()