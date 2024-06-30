# Rohlik-Orders-Forecasting-Challenge
Kaggle competition

## Logging

### In case of using `neptune`
```python
import neptune
run = neptune.init_run(project='srushanthbaride/Rohlik-Orders-Forecasting-Challenge')
```

### In case of using `MLFlow `

## Overview
[Rohlik Group](https://www.rohlik.group/), a leading European e-grocery innovator, is revolutionising the food retail industry. We operate across 11 warehouses in the Czech Republic, Germany, Austria, Hungary, and Romania.

Our challenge focuses on predicting the number of orders (grocery deliveries) at selected warehouses for the next 60 days.

## Description
### Why This Matters
Accurate order forecasts are crucial for planning process, impacting workforce allocation, delivery logistics, inventory management, and supply chain efficiency. By optimizing forecasts, we can minimize waste and streamline operations, making our e-grocery services more sustainable and efficient.

### Your Impact
Your participation in this challenge will directly contribute to Rohlik mission of sustainable and efficient e-grocery delivery. Your insights will help us enhance customer service and achieve a greener future.

## Evaluation
Submissions are evaluated on Mean Absolute Percentage Error between the predicted orders and the actual orders.

## Submission File
For each ID in the test set, you must predict the number of orders. The file should contain a header and have the following format:

| ID                  | ORDERS |
|---------------------|--------|
| Prague_1_2024-03-16 | 5000   |
| Prague_1_2024-03-17 | 5000   |
| Prague_1_2024-03-18 | 5000   |
| etc.                |        |

## Prizes
### Leaderboard prizes

- 1st place - $5,000
- 2nd place - $5,000
- 3rd place - $2,000

## Timeline
__August 9, 2024__ - First Submission deadline. Your team must make its first submission by this deadline.
__August 9, 2024__ - Team Merger deadline. This is the last day you may merge with another team.
__August 23, 2024__ - Final submission deadline
All deadlines are at 11:59 PM CET on the corresponding day unless otherwise noted. The organizers reserve the right to update the contest timeline if they deem it necessary.

## Citation

[MichalKecera. (2024). *Rohlik Orders Forecasting Challenge*. Kaggle.](https://kaggle.com/competitions/rohlik-orders-forecasting-challenge)
