# Coffee Retail Analysis

Comprehensive end-to-end analytics project for coffee retail sales using Python and Jupyter.

This project covers:

- Data loading (local + Kaggle-aware path handling)
- Data quality checks (nulls, duplicates, type validation)
- Data cleaning and table merging (orders + customers + products)
- Feature engineering (time, seasonality, profit, pricing buckets)
- Exploratory data analysis (products, trends, geography, distributions)
- Loyalty and statistical analysis
- Customer segmentation (RFM + clustered view)
- Business insight generation and exportable outputs

## Repository Structure

- `coffee_analysis_complete.ipynb` — Main notebook with full workflow
- `Dataset/COFFEE_ANALISYS_PROJECT.xlsx` — Source dataset (Excel with `orders`, `customers`, `products`)
- `All images of VIsualizations/` — Exported charts from notebook runs
- `requirements.txt` — Python dependencies

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run:

- `coffee_analysis_complete.ipynb`

## Outputs Produced

When run end-to-end, the notebook can generate:

- Visualization PNG files (product, time-series, geography, RFM, etc.)
- `coffee_clean_master.csv`
- `coffee_rfm_segments.csv`
- `coffee_rfm_advanced_segments.csv` (when advanced clustering section is executed)

## Notes

- If running outside this folder structure, adjust dataset path candidates in the data loading section.
- The workbook name is expected as: `COFFEE_ANALISYS_PROJECT.xlsx`.

## Author

**Sajjad Ali Shah**  
ML Engineer & Data Scientist

- LinkedIn: https://www.linkedin.com/in/sajjad-ali-shah/
- GitHub: https://github.com/SajjadKhanYousafzai
- Kaggle: https://www.kaggle.com/sajjadalishah
