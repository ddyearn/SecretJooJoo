import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 우리가 사용할 자산 리스트 생성
assets = []

# 이 부분은 퀀티와이즈나 와이즈리포트를 사용해서 얻어오는 데이터를 조금 가공하면 될것같음
''' 
# 모듈경로 지정
import sys
sys.path.append(r'C:\Users\kwonk\Downloads\개인 프로젝트\juno1412-1\증권데이터분석\DB_API')

# 모듈 가져오기
import Analyzer

mk = Analyzer.MarketDB()

# 종목별 일별 시세 dataframe 생성
df = pd.DataFrame()
for i in assets:
    df[i] = mk.get_daily_price(i, '2018-10-13', '2022-12-23')['close']

# 데이터를 토대로 종목별, 일간 수익률, 연간수익률, 일간리스크, 연간리스크를 구하기
# 5종목 일간 변동률
daily_ret = df.pct_change()
# 5종목 1년간 변동률 평균(252는 미국 1년 평균 개장일)
annual_ret = daily_ret.mean() * 252

# 5종목 연간 리스크 = cov()함수를 이용한 일간변동률 의 공분산
daily_cov = daily_ret.cov()
# 5종목 1년간 리스크(252는 미국 1년 평균 개장일)
annual_cov = daily_cov * 252
'''


# 시가총액 5순위 주식의 비율을 다르게 해 20,000개 포트폴리오 생성
# 1. 수익률, 리스크, 비중 list 생성
# 수익률 = port_ret
# 리스크 = port_risk
# 비  중 = port_weights
port_ret = []
port_risk = []
port_weights = []
# 샤프지수 추가
shape_ratio = []

for i in range(20000):
    # 2. 랜덤 숫자 4개 생성 - 랜덤숫자 4개의 합 = 1이되도록 생성
    weights = np.random.random(len(assets))
    weights /= np.sum(weights)
    
    # 3. 랜덤 생성된 종목뵹 비중 배열과 종목별 연간 수익률을 곱해 포트폴리오의 전체 수익률(returns)를 구한다.
    returns = np.dot(weights, annual_ret)

    # 4. 종목별 연간공분산과 종목별 비중배열 곱하고, 다시 종목별 비중의 전치로 곱한다.
    # 결과값의 제곱근을 sqrt()함수로 구하면 해당 포트폴리오 전체 risk가 구해진다. 
    risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))

    # 5. 20,000개 포트폴리오의 수익률, 리스크, 종목별 비중을 각각 리스트에 추가한다.
    port_ret.append(returns)
    port_risk.append(risk)
    port_weights.append(weights)
    shape_ratio.append(returns/risk)

# 포트폴리오 결과에 샤프지수 추가
portfolio = {'Returns' : port_ret, 'Risk' : port_risk, 'Shape' : shape_ratio}
for j, s in enumerate(assets):
    # 6. portfolio 4종목의 가중치 weights를 1개씩 가져온다.
    portfolio[s] = [weight[j] for weight in port_weights]

# 7. 최종 df는 시총 상위 5종목의 보유 비중에 따른 risk와 예상 수익률을 확인할 수 있다.
df = pd.DataFrame(portfolio)
df = df[['Returns', 'Risk', 'Shape'] + [s for s in assets]]


# 8. 샤프지수로 위험단위당 예측 수익률이 가장 높은 포트폴리오 구하기
# 샤프지수 칼럼에서 가장 높은 샤프지수 구하기
max_shape = df.loc[df['Shape'] == df['Shape'].max()]

# 리스크칼럼에서 가장 낮은 리스크 구하기
min_risk = df.loc[df['Risk'] == df['Risk'].min()]

# 샤프지수 그래프 그리기
df.plot.scatter(x='Risk', y='Returns', c='Shape', cmap='viridis', edgecolors='k', figsize=(10,8), grid=True)
plt.scatter(x=max_shape['Risk'], y=max_shape['Returns'], c='r', marker='X', s=300)
plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='r', marker='X', s=200)
plt.title('Portfolio Optimization')
plt.xlabel('Risk')
plt.ylabel('Expected Return')
plt.show()