# Pet Shop Data Science Case

## Business Context

This project explores how a multi-channel pet shop can use customer and transaction data to improve demand planning, customer retention, segmentation, and product recommendations. The business operates physical stores and offers services such as veterinary care and grooming, alongside online sales.

The team wants to make better use of purchase history to understand demand, identify different customer behaviors, and support more relevant retention and recommendation actions.

## Important Note About the Data

The datasets included with this project are **samples for demonstration**. They are not the original business data. The included sample generator creates synthetic pet-retail data for reproducible analysis; sample rows and aggregate patterns should not be interpreted as actual company activity.

Consequently, all charts, cluster profiles, churn rates, forecasts, recommendations, and evaluation metrics shown in the notebooks are specific to the sample data and may differ when the original data is used. They are illustrative results, not business estimates or validated production performance. The available sample tables primarily contain customer, product, and transaction information; they do not include browsing events or monetary fields such as revenue, cost, or margin.

## Business Questions

1. **Demand forecasting:** What is the expected daily demand for dog products over the next six months?
2. **Customer segmentation:** Which distinct purchase-behavior groups can be identified, and how do they differ? Monetary value or profitability comparisons require revenue and margin data that are not present in the included samples.
3. **Churn and retention:** Given the project's 90-day inactivity definition, which customers appear most likely to become inactive and could be prioritized for retention actions?
4. **Product recommendations:** Which products can be recommended to customers based on their purchase history? Browsing-based recommendations require browsing data, which is not included in the sample tables.

## Project Structure

- `notebooks/01.saleforecast.ipynb` — demand exploration and forecasting
- `notebooks/02.customer_segmentation.ipynb` — behavioral customer segmentation
- `notebooks/03.churn_recommendation.ipynb` — churn labeling and classification
- `notebooks/04.product_recomendation.ipynb` — collaborative filtering and basket analysis
- `src/` — shared data, feature, evaluation, training, and sample-generation utilities
- `data/sample/` — included sample datasets

## Running the Sample Data Generator

From the repository root, generate a deterministic synthetic dataset with:

```bash
python -m src.sample --output-dir data/sample
```

The generator accepts `--seed`, `--customers`, `--products`, and `--orders` options. Changing the seed or dataset size can also change the analytical results.
