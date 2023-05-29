import bt
import pandas as pd
from datetime import datetime
from datetime import timedelta
import re
import matplotlib.pyplot as plt

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

    df = pd.read_excel(io='./20171101_20230501.xlsx',
                sheet_name='Sheet2',
                usecols=f'A, {colc}',
                index_col = 0,
                skiprows=13)
    
    return df 

assets = ['A069500', 'A143850', 'A238720', 'A148070', 'A182490', 'A132030', 'A139660', 'A138230', 'A130730']

price = pd.DataFrame()
for i in assets:
    price[i] = get_daily_price(i, '2017-11-01', '2023-05-01')

# 전체 자산 동일비중, 매월 초 리밸런싱
strategy = bt.Strategy("Equal Weight", [
    bt.algos.SelectAll(),
    bt.algos.WeighEqually(),
    bt.algos.RunMonthly(run_on_first_date=False, run_on_end_of_period=False, run_on_last_date=True),
    bt.algos.Rebalance()
])


# 백테스트 생성
backtest = bt.Backtest(strategy, price, name='Equal Weight', commissions=lambda q, p: abs(q) * p * 0.00015)

# # 백테스트 실행
# result = bt.run(backtest)


# 최대샤프 비중 설정
mps = bt.Strategy('My Portfolio', [
    bt.algos.SelectAll(),
    bt.algos.WeighSpecified(A069500=0.10392766, A143850=0.02571382, A238720=0.09019623, A148070=0.06467682, A182490=0.16923984, A132030=0.10300958, A139660=0.00123652, A138230=0.19350187, A130730=0.24849766),
    bt.algos.RunMonthly(run_on_first_date=False, run_on_end_of_period=False, run_on_last_date=True),
    bt.algos.Rebalance()
])
mps_backtest = bt.Backtest(mps, price, name='My Portfolio', commissions=lambda q, p: abs(q) * p * 0.00015)
# aw_result = bt.run(aw_backtest)


# 최소위험 비중 설정
mpr = bt.Strategy('My Portfolio MinRisk', [
    bt.algos.SelectAll(),
    bt.algos.WeighSpecified(A069500=0.10375375, A143850=0.06658983, A238720=0.03783368, A148070=0.17780594, A182490=0.09489301, A132030=0.07447156, A139660=0.00105201, A138230=0.18866943, A130730=0.25493078),
    bt.algos.RunMonthly(run_on_first_date=False, run_on_end_of_period=False, run_on_last_date=True),
    bt.algos.Rebalance()
])
mpr_backtest = bt.Backtest(mpr, price, name='My Portfolio MinRisk', commissions=lambda q, p: abs(q) * p * 0.00015)



# 결과
result = bt.run(backtest, mps_backtest, mpr_backtest)

# 그래프 그리기
# result.plot(figsize=(10, 6), title='My Portfolio', legend=False)
# plt.show()
result.prices.iloc[201:, ].rebase().to_drawdown_series().plot(
    figsize=(10, 6))
plt.show()

