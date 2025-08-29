#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 19:55:19 2025

@author: rehangupta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 22:19:16 2025

@author: rehangupta
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 15:50:07 2025

@author: rehangupta
"""

from binance.client import Client as bnb_client
from datetime import datetime
import pandas as pd 
import numpy as np 
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
#client = bnb_client()
###  if you're in the US, use: 
client = bnb_client(tld='US')#" here instead
def drawdown(px):
    return (px / px.expanding(min_periods=1).max() - 1)
def get_binance_px1(symbol,freq,start_ts = '2021-08-01',end_ts='2024-08-01'):
    data = client.get_historical_klines(symbol,freq,start_ts,end_ts)
    columns = ['open_time','open','high','low','close','volume','close_time','quote_volume',
    'num_trades','taker_base_volume','taker_quote_volume','ignore']

    data = pd.DataFrame(data,columns = columns)
    
    # Convert from POSIX timestamp (number of millisecond since jan 1, 1970)
    data['open_time'] = data['open_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    data['close_time'] = data['close_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    return data 
def get_binance_px2(symbol,freq,start_ts = '2024-08-01',end_ts='2025-08-01'):
    data = client.get_historical_klines(symbol,freq,start_ts,end_ts)
    columns = ['open_time','open','high','low','close','volume','close_time','quote_volume',
    'num_trades','taker_base_volume','taker_quote_volume','ignore']

    data = pd.DataFrame(data,columns = columns)
    
    # Convert from POSIX timestamp (number of millisecond since jan 1, 1970)
    data['open_time'] = data['open_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    data['close_time'] = data['close_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    return data 

univ = [
    'BTCUSDT',   # Bitcoin
    'ETHUSDT',   # Ethereum
    'XRPUSDT',   # Ripple
    'BCHUSDT',   # Bitcoin Cash
    'LTCUSDT',   # Litecoin
    'EOSUSDT',   # EOS
    'ADAUSDT',   # Cardano
    'XLMUSDT',   # Stellar
    'TRXUSDT',   # TRON
    'ETCUSDT',   # Ethereum Classic
    'LINKUSDT',  # Chainlink
    'DASHUSDT',  # Dash
    'IOTAUSDT',  # IOTA
    'ZECUSDT',   # Zcash
    'DOGEUSDT',  # Dogecoin
    'ICXUSDT',     #ICON
    'BATUSDT', #Basic Attention Token
    'VETUSDT', #VeChain
    'OMGUSDT', # OMG Network
    'ALGOUSDT', #Algorand
    'WAVESUSDT', #Waves
    'XTZUSDT', #Tezos
    'ATOMUSDT',# Cosmos
    'MKRUSDT', #Maker
    'SOLUSDT',#Solana
    'USDCUSDT',#USDC
    'SUIUSDT',#Sui
    'SHIBUSDT',#Shiba Inu
    'HBARUSDT',#Hedera Hashgraph
    'DOTUSDT',#Polkadot
    'UNIUSDT',#Uniswap
    'DAIUSDT',#Dai
    'NEARUSDT',#NEAR Protocol
    'AAVEUSDT',#Aave
    'ICPUSDT',#Internet Computer
    'APTUSDT',# Aptos
    'KNCUSDT',# Kyber Network
    'ZILUSDT',#Zilliqa
    'LSKUSDT',#Lisk
    'RENUSDT',#Ren
    'IOSTUSDT',#IOST
]

start_date_required = datetime(2021, 8, 1)
filtered_univ = []
px = {}
freq = '4h'
px = {}
px_test={}
for symbol in univ:
    try:
        data = get_binance_px1(symbol, freq)
        first_date = data['open_time'].min()
        if pd.isna(first_date):
            print(f"Skipping {symbol} — no data returned")
            continue
        if first_date <= start_date_required:
            px[symbol] = data.set_index('open_time')['close']
            filtered_univ.append(symbol)
        else:
            print(f"Skipping {symbol} — data starts {first_date.date()}")
    except Exception as e:
        print(f"Skipping {symbol} — error: {e}")

# Replace the original universe if you want:
univ = filtered_univ
print("Final universe:", univ)

freq = '4h'
px = {}
for x in univ:
    data = get_binance_px1(x,freq)
    px[x] = data.set_index('open_time')['close']
px = pd.DataFrame(px).astype(float)
px = px.reindex(pd.date_range(px.index[0],px.index[-1],freq=freq))
ret = px.pct_change()
Market=ret["BTCUSDT"]
for x in univ:
    data_test = get_binance_px2(x,freq)
    px_test[x] = data_test.set_index('open_time')['close']
px_test = pd.DataFrame(px_test).astype(float)
px_test = px_test.reindex(pd.date_range(px_test.index[0],px_test.index[-1],freq=freq))
ret_test = px_test.pct_change()
Market_test=ret_test["BTCUSDT"]
ROLLING_WINDOW=60 #Fixed rolling window of 60 bars for testing
normalized_prices_test=np.log(px_test)
corr_series={}
spread={}
beta_dict={}
alpha_dict={}
column_list1=list(ret.columns)
column_list1.remove("BTCUSDT")
#Training
for i in column_list1:
    list1=[]
    for j in column_list1:
        if j!=i:
            list1.append(j)
    px2=px[list1]
    series1=px[i].corr(px2.mean(1))
    corr_series[i]=series1
basket_corr=pd.Series(corr_series)
normalized_prices=np.log(px)

for i in column_list1:
    list1=[]
    for j in column_list1:
        if j!=i:
            list1.append(j)
    mean_price=(normalized_prices[list1]).mean(1)
    Y=normalized_prices[i]
    temp_df=pd.concat([mean_price,Y],axis=1).dropna()
    temp_df.columns=["Basket","Individual_Coin"]
    X=temp_df["Basket"]
    Y=temp_df["Individual_Coin"]
    X=sm.add_constant(X)
    model=sm.OLS(Y,X).fit()
    beta_dict[i]=model.params["Basket"]
    alpha_dict[i]=model.params["const"]
column_list2=list(ret.columns)
column_list2.remove("BTCUSDT")
for i in column_list1:
    if(beta_dict[i]<0.5 or beta_dict[i]>1.5 or corr_series[i]<0.5):
        column_list2.remove(i)
        beta_dict.pop(i)
        alpha_dict.pop(i)
for i in column_list2:
    list1=[]
    for j in column_list2:
        if j!=i:
            list1.append(j)
    mean_price=(normalized_prices[list1]).mean(1)
    Y=normalized_prices[i]
    temp_df=pd.concat([mean_price,Y],axis=1).dropna()
    temp_df.columns=["Basket","Individual_Coin"]
    X=temp_df["Basket"]
    Y=temp_df["Individual_Coin"]
    X=sm.add_constant(X)
    model=sm.OLS(Y,X).fit()
    spread[i]=model.resid
spread_df=pd.DataFrame(spread)
beta_series=pd.Series(beta_dict)
z_scores=(spread_df-spread_df.mean(0))/spread_df.std(0)
z_scores2=z_scores.where(np.abs(z_scores)>0.5)
z_scores2=z_scores2*-1
z_scores2=z_scores2*0.5
sum_zscores=z_scores2.abs().sum(axis=1)
sum_zscores=sum_zscores.where(sum_zscores!=0,1e-10)
weights=z_scores2.div(sum_zscores,axis=0)
portfolio_ret=(weights.shift(1)*ret[column_list2])
portfolio_ret=portfolio_ret.fillna(0)
portfolio_ret=portfolio_ret.sum(1)
shifted_weights=weights.shift(1).fillna(0)
turnover=((weights.fillna(0))-shifted_weights).abs().sum(1)
tcosts=0.002*turnover
net_portfolio_ret=portfolio_ret-tcosts
mean_portfolio_ret=net_portfolio_ret.mean(0)*2190
std_portfolio_ret=net_portfolio_ret.std(0)*np.sqrt(2190)
sharpe_portfolio_ret_train=mean_portfolio_ret/std_portfolio_ret
temp_df=pd.concat([Market,net_portfolio_ret],axis=1).dropna()
temp_df.columns=["Market","PortfolioReturn"]
Y=temp_df["PortfolioReturn"]
X=temp_df["Market"]
X=sm.add_constant(X)
model2=sm.OLS(Y,X).fit()
beta_portfolio_ret_train=model2.params["Market"]
alpha_ret_train=model2.params["const"]
t_stat_alpha_train = model2.tvalues['const']
cum_3 = (1 + net_portfolio_ret).cumprod()
dd = drawdown(cum_3)
max_dd_train = dd.min()
training_cumulative_series=cum_3
print(max_dd_train)
print(beta_portfolio_ret_train)
print(alpha_ret_train)
print(t_stat_alpha_train)
print(sharpe_portfolio_ret_train)
#Testing
spread_test={}
for i in column_list2:
    list1=[]
    for j in column_list2:
        if j!=i:
            list1.append(j)
    mean_price_test=(normalized_prices_test[list1]).mean(1)
    Y=normalized_prices_test[i]
    temp_df=pd.concat([mean_price_test,Y],axis=1).dropna()
    temp_df.columns=["Basket","Individual_Coin"]
    X=temp_df["Basket"]
    Y=temp_df["Individual_Coin"]
    spread_test[i]=Y-(beta_dict[i]*X)-alpha_dict[i]
spread_df_test=pd.DataFrame(spread_test)
beta_series_testing=pd.Series(beta_dict)
z_scores_test=(spread_df_test-spread_df_test.rolling(window=ROLLING_WINDOW).mean())/spread_df_test.rolling(window=60).std(0)
z_scores2_test=z_scores_test.where(np.abs(z_scores_test)>0.5)
z_scores2_test=z_scores2_test*-1
z_scores2_test=z_scores2_test*0.5
sum_zscores_test=z_scores2_test.abs().sum(axis=1)
sum_zscores_test=sum_zscores_test.where(sum_zscores_test!=0,1e-10)
weights_test=z_scores2_test.div(sum_zscores_test,axis=0)
portfolio_ret_test=(weights_test.shift(1)*ret_test[column_list2])
portfolio_ret_test=portfolio_ret_test.fillna(0)
portfolio_ret_test=portfolio_ret_test.sum(1)
shifted_weights_test=weights_test.shift(1).fillna(0)
turnover_test=(weights_test-shifted_weights_test).abs().sum(1)
tcosts_test=0.002*turnover_test
net_portfolio_ret_test=portfolio_ret_test-tcosts_test
mean_portfolio_ret_test=net_portfolio_ret_test.mean(0)*2190
std_portfolio_ret_test=net_portfolio_ret_test.std(0)*np.sqrt(2190)
sharpe_portfolio_ret_test=mean_portfolio_ret_test/std_portfolio_ret_test
temp_df=pd.concat([Market_test,net_portfolio_ret_test],axis=1).dropna()
temp_df.columns=["Market","PortfolioReturn"]
Y=temp_df["PortfolioReturn"]
X=temp_df["Market"]
X=sm.add_constant(X)
model2=sm.OLS(Y,X).fit()
beta_portfolio_ret_test=model2.params["Market"]
alpha_ret_test=model2.params["const"]
t_stat_alpha_test = model2.tvalues['const']
cum_3 = (1 + net_portfolio_ret_test).cumprod()
dd = drawdown(cum_3)
max_dd_test = dd.min()
testing_cumulative_series=cum_3
cumulative_return=pd.DataFrame({"Training Cumulative Returns":training_cumulative_series,"Testing Cumulative Returns":testing_cumulative_series})
cumulative_plot=cumulative_return.plot(title="Cumulative Returns over time",xlabel="Time",ylabel="Cumulative Return")
print(max_dd_test)
print(beta_portfolio_ret_test)
print(alpha_ret_test)
print(t_stat_alpha_test)
print(sharpe_portfolio_ret_test)