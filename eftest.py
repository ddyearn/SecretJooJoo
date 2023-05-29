import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import re
import matplotlib.pyplot as plt

# 우리가 사용할 자산 리스트 생성
assets = ['A069500', 'A143850', 'A238720', 'A148070', 'A182490', 'A132030', 'A139660', 'A138230', 'A130730']

def get_daily_price(code, start_date=None, end_date=None):
    """ 
        KRX 종목의 일별 시세를 데이터프레임 형태로 반환
        - code       : KRX 종목코드('005930') 또는 상장기업명('삼성전자')
        - start_date : 조회 시작일('2020-01-01'), 미입력 시 1년 전 오늘
        - end_date   : 조회 종료일('2020-12-31'), 미입력 시 오늘 날짜
    """
    if start_date is None:
        one_year_ago = datetime.today() - timedelta(days=365)
        start_date = one_year_ago.strftime('%Y-%m-%d')
        print("start_date is initialized to '{}'".format(start_date))
    else:
        start_lst = re.split('\D+', start_date)
        if start_lst[0] == '':
            start_lst = start_lst[1:]
        start_year = int(start_lst[0])
        start_month = int(start_lst[1])
        start_day = int(start_lst[2])
        if start_year < 1900 or start_year > 2200:
            print(f"ValueError: start_year({start_year:d}) is wrong.")
            return
        if start_month < 1 or start_month > 12:
            print(f"ValueError: start_month({start_month:d}) is wrong.")
            return
        if start_day < 1 or start_day > 31:
            print(f"ValueError: start_day({start_day:d}) is wrong.")
            return
        start_date=f"{start_year:04d}-{start_month:02d}-{start_day:02d}"

    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
        print("end_date is initialized to '{}'".format(end_date))
    else:
        end_lst = re.split('\D+', end_date)
        if end_lst[0] == '':
            end_lst = end_lst[1:] 
        end_year = int(end_lst[0])
        end_month = int(end_lst[1])
        end_day = int(end_lst[2])
        if end_year < 1800 or end_year > 2200:
            print(f"ValueError: end_year({end_year:d}) is wrong.")
            return
        if end_month < 1 or end_month > 12:
            print(f"ValueError: end_month({end_month:d}) is wrong.")
            return
        if end_day < 1 or end_day > 31:
            print(f"ValueError: end_day({end_day:d}) is wrong.")
            return
        end_date = f"{end_year:04d}-{end_month:02d}-{end_day:02d}"
    
    if code in assets:
        colc = chr(assets.index(code) + 66)
    else:
        print(f"ValueError: Code({code}) doesn't exist.")

    df = pd.read_excel(io='./daily_price_data.xlsx',
                sheet_name='Sheet2',
                usecols=f'A, {colc}',
                index_col = 0,
                skiprows=13)
    
    return df 

assets = ['A069500', 'A143850', 'A238720', 'A148070', 'A182490', 'A132030', 'A139660', 'A138230', 'A130730']
def weight_const(w):
    '''
        A069500 상한선 40%, 10% 0
        A143850 상한선 20% 1
        A238720 상한선 20% 2
        A143850 + A238720 상한선 40% 하한선 10% 1,2
        A148070 상한선 50% 3
        A182490 상한선 40% 하한선 5% 4
        A148070 + A182490 상한선 60% 하한선 20% 3,4
        A132030 상한선 15%  하한선 5% 5
        A139660 상한선 20% 6
        A138230 상한선 20% 7
        A139660 + A138230 상한선 20% 6,7
        A130730 상한선 50% 하한선 없음 8
    '''
    if (w[0] > 0.4) | (w[0] < 0.1) :
        return False
    if w[1] > 0.2:
        return False
    if w[2] > 0.2 :
        return False
    if (w[1]+w[2] > 0.4) | (w[1]+w[2] < 0.1) :
        return False
    if w[3] > 0.5:
        return False
    if (w[4] > 0.4) | (w[4] < 0.05) :
        return False
    if (w[3]+w[4] > 0.6) | (w[3]+w[4] < 0.2) :
        return False
    if (w[5] > 0.15) | (w[5] < 0.05) :
        return False
    if w[6] > 0.2:
        return False
    if w[7] > 0.2 :
        return False
    if w[6]+w[7] > 0.2 :
        return False
    if w[8] > 0.5:
        return False
    return True

        # 종목별 일별 시세 dataframe 생성
df = pd.DataFrame()
for i in assets:
    df[i] = get_daily_price(i, '2018-06-01', '2023-05-24')


# 데이터를 토대로 종목별, 일간 수익률, 연간수익률, 일간리스크, 연간리스크를 구하기
# 5종목 일간 변동률
daily_ret = df.pct_change()
# 5종목 1년간 변동률 평균(252는 미국 1년 평균 개장일)
annual_ret = daily_ret.mean() * 252

# 5종목 연간 리스크 = cov()함수를 이용한 일간변동률 의 공분산
daily_cov = daily_ret.cov()
# 5종목 1년간 리스크(252는 미국 1년 평균 개장일)
annual_cov = daily_cov * 252


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

kospi200 = []
snp500 = []
n225 = []
ktb = []
sthy = []
g = []
usdx = []
usd = []
mmf = []

def ef():
    for i in range(200000):
        # 2. 랜덤 숫자 4개 생성 - 랜덤숫자 4개의 합 = 1이되도록 생성
        weights = np.random.random(len(assets))
        weights /= np.sum(weights)

        if weight_const(weights) == False:
            continue
        
        # 3. 랜덤 생성된 종목뵹 비중 배열과 종목별 연간 수익률을 곱해 포트폴리오의 전체 수익률(returns)를 구한다.
        returns = np.dot(weights, annual_ret)

        # 4. 종목별 연간공분산과 종목별 비중배열 곱하고, 다시 종목별 비중의 전치로 곱한다.
        # 결과값의 제곱근을 sqrt()함수로 구하면 해당 포트폴리오 전체 risk가 구해진다. 
        risk = np.sqrt(np.dot(weights.T, np.dot(annual_cov, weights)))

        # 5. 200,000개 포트폴리오의 수익률, 리스크, 종목별 비중을 각각 리스트에 추가한다.
        port_ret.append(returns)
        port_risk.append(risk)
        port_weights.append(weights)
        shape_ratio.append(returns/risk)

    # 포트폴리오 결과에 샤프지수 추가
    portfolio = {'Returns' : port_ret, 'Risk' : port_risk, 'Shape' : shape_ratio}
    for j, s in enumerate(assets):
        # 6. portfolio 9종목의 가중치 weights를 1개씩 가져온다.
        portfolio[s] = [weight[j] for weight in port_weights]

    # 7. 최종 df는 9종목의 보유 비중에 따른 risk와 예상 수익률을 확인할 수 있다.
    df = pd.DataFrame(portfolio)
    df = df[['Returns', 'Risk', 'Shape'] + [s for s in assets]]

    # print(df)

    # 8. 샤프지수로 위험단위당 예측 수익률이 가장 높은 포트폴리오 구하기
    # 샤프지수 칼럼에서 가장 높은 샤프지수 구하기
    max_shape = df.loc[df['Shape'] == df['Shape'].max()]

    # 리스크칼럼에서 가장 낮은 리스크 구하기
    min_risk = df.loc[df['Risk'] == df['Risk'].min()]

    kospi200.append(min_risk['A069500'].values)
    snp500.append(min_risk['A143850'].values)
    n225.append(min_risk['A238720'].values)
    ktb.append(min_risk['A148070'].values)
    sthy.append(min_risk['A182490'].values)
    g.append(min_risk['A132030'].values)
    usdx.append(min_risk['A139660'].values)
    usd.append(min_risk['A138230'].values)
    mmf.append(min_risk['A130730'].values)

# rc = 100
# for i in range(rc):
#     ef()

# kospi200_rate = sum(kospi200) / rc
# snp500_rate = sum(snp500) / rc
# n225_rate = sum(n225) / rc
# ktb_rate = sum(ktb) / rc
# sthy_rate = sum(sthy) / rc
# g_rate = sum(g) / rc
# usdx_rate = sum(usdx) / rc
# usd_rate = sum(usd) / rc
# mmf_rate = sum(mmf) / rc
# print(kospi200_rate)
# print(snp500_rate)
# print(n225_rate)
# print(ktb_rate)
# print(sthy_rate)
# print(g_rate)
# print(usdx_rate)
# print(usd_rate)
# print(mmf_rate)
# max_return = df.loc[df['Returns'] == df['Returns'].max()]
# target_return = df[(df['Returns'] >= 0.055) & (df['Returns'] <= 0.06) & (df['Risk'] <= 0.06)]

# print("max_shape")
# print(max_shape)
# print("min_risk")
# print(min_risk)
# print(max_shape['A069500'].values)
# print("max_return")
# print(max_return)
# print("target_return")
# print(target_return)



# 비중 직접 설정
print("비중 직접 설정")
# ['A069500', 'A143850', 'A238720', 'A148070', 'A182490', 'A132030', 'A139660', 'A138230', 'A130730']
my_weights = [0.10375375, 0.06658983, 0.03783368, 0.17780594, 0.09489301, 0.07447156, 0.00105201, 0.18866943, 0.25493078]
my_weights = np.array(my_weights, dtype=np.float64)
# 3. 랜덤 생성된 종목뵹 비중 배열과 종목별 연간 수익률을 곱해 포트폴리오의 전체 수익률(returns)를 구한다.
returns = np.dot(my_weights, annual_ret)

# 4. 종목별 연간공분산과 종목별 비중배열 곱하고, 다시 종목별 비중의 전치로 곱한다.
# 결과값의 제곱근을 sqrt()함수로 구하면 해당 포트폴리오 전체 risk가 구해진다. 
risk = np.sqrt(np.dot(my_weights.T, np.dot(annual_cov, my_weights)))

# 5. 20,000개 포트폴리오의 수익률, 리스크, 종목별 비중을 각각 리스트에 추가한다.
print("return")
print(returns)
print("risk")
print(risk)
print("shape")
print(returns/risk)

#Plot minimum variance frontier (최소 분산 곡선 그래프)
#Random portfolios (무작위 포트폴리오)
# means = df['Returns'].values
# stds = df['Risk'].values
 
# #Minimum varaince portfolios (최소분산 포트폴리오)
# opt_returns = min_risk['Returns']
# opt_stds = min_risk['Risk']
 
# # fig = plt.figure(figsize=(15,8))
# df.plot.scatter(x='Risk', y='Returns', c='Shape', cmap='viridis', edgecolors='k', figsize=(10,8), grid=True)
# plt.plot(stds, means, 'o')
# plt.ylabel('mean',fontsize=12)
# plt.xlabel('std',fontsize=12)
# plt.plot(opt_stds, opt_returns, 'y-o')
# # plt.xlim(0.013,0.025)
# # plt.ylim(0.001,0.0024)
# plt.title('Minimum variance frontier for risky assets',fontsize=15)
# plt.show()


# 샤프지수 그래프 그리기
# df.plot.scatter(x='Risk', y='Returns', c='Shape', cmap='viridis', edgecolors='k', figsize=(10,8), grid=True)
# plt.scatter(x=max_shape['Risk'], y=max_shape['Returns'], c='r', marker='X', s=300)
# plt.scatter(x=min_risk['Risk'], y=min_risk['Returns'], c='r', marker='X', s=200)
# plt.title('Portfolio Optimization')
# plt.xlabel('Risk')
# plt.ylabel('Expected Return')
# plt.show()

# plt.scatter(df['Risk'], df['Returns'], c=np.array(df['Returns']) / np.array(df['Risk']), marker='.')
# plt.plot()

# trets = np.linspace(0.0, 0.25, 50)
# tvols = []

# for tret in trets:
#     cons = ({'type': 'eq', 'fun': lambda x: statistics(x)[0] - tret}, {'type':'eq', 'fun': lambda x: np.sum(x) - 1})
#     res = opt.minimize(min_func_port, 9*[1./0], method='SLSQP', bounds = bnds, constraints=cons)
#     tvols.append(res['fun'])

# tvols = np.array(tvols)
# ind = np.argmin(tvols)
# evols = tvols[ind:]
# erets = trets[ind:]

# plt.scatter(df['Risk'], df['Returns'], c=np.array(df['Returns']) / np.array(df['Risk']), marker='.')
# plt.plot(evols, erets, 'g', lw=4.0)
# plt.show()