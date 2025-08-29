# Quant-Mean-reversion-statistical-arbitrage-project

                                                                                                
The statistical arbitrage strategy I used is a correlation strategy that profits of mean-reversion and is similar to pairs trading. I used binance API to retrieve the price data of 46 coins for the last 4 years. The goal of the strategy is to find correlations between a crypto coin and a basket of coins. While the coin and the basket usually move together, they occasionally diverge and revert to their means. When they diverge, we can long the asset that has gone down expecting its price to increase and short the asset that has gone up expecting its price to fall. 
The strategy is trained on 4-hour prices between 1st August 2021 to 31st 2024 and tested on 4-hour prices between 1st August 2024 to 31st July. We do not use training data from the peak COVID-19 pandemic period because that was a black swan event with high volatility unlikely to be replicated in the future and hence any model trained on that data would not be predictive. 
While measuring Beta and Alpha of a strategy in the traditional stock market we assume SPY(S&P 500) to be the market benchmark. The same way in cryptocurrency I assume BTC(Bitcoin) to be the market benchmark. 
                                                                               
                                                                                                   
The spread between a correlated coin and a basket of coins is computed to be the residue of an OLS (Ordinary Least Squares) model between the coin price and the mean basket price. We then calculated the z_scores of the spread over the training time period. The z_scores are then multiplied by -1 since we exploit reversion. I then multiply the z_scores by 0.5 to control aggressiveness. The 0.5 value is a hyperparameter. 
Hyperparameters are constant manually set variables that control how the model trains and learns. Using the OLS model I calculate the beta between each coin and the basket and the alpha between each coin and the basket. I remove the coins (betas and alphas) that have a low correlation (below 0.5) and have a beta of less than 0.5 or greater than 1.5 to their basket as that could indicate a forced relationship.  
                                                                                             
                                                                                                  
Parameters are variables that are learned during training. My strategy has a fixed executional cost (slippage and commission) of 20bps. To reduce my transactional costs and turnover I use the where function in the Pandas library to remove weak z_scores and preserve the strong ones increasing alpha and trading on significant spreads. I also test 3 different weighting methods along with different z_score filters to train my model as much as possible so that it can give good results in an unseen timeframe(testing). 



                                  
 

<img width="451" height="691" alt="image" src="https://github.com/user-attachments/assets/1ca4d593-0672-4dbd-ab3d-e6baa610b10e" />
