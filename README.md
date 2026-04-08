# Falcon 9 First Stage Landing Prediction

Capstone project for the IBM Data Science Professional Certificate. 
The goal: predict whether a Falcon 9 first stage will successfully land after launch, using historical launch data from SpaceX.

SpaceX lists Falcon 9 launches at $62M, significantly cheaper than competitors charging upward of $165M. The cost difference comes largely from booster reusability. If you can predict a landing outcome in advance, you can estimate the actual cost of a launch. That's the practical framing behind this project.

---

## Project Structure

| Notebook | What it does |
|----------|-------------|
| `1-data-collection-api.ipynb` | Pulls launch data from the SpaceX REST API, enriches it with booster, payload, and launchpad details via nested API calls |
| `2-webscraping.ipynb` | Scrapes Falcon 9 and Falcon Heavy launch records from Wikipedia as a secondary data source |
| `3-data-wrangling.ipynb` | Cleans and labels the dataset — converts raw landing outcome strings into a binary `Class` column for supervised learning |
| `4-eda-sql.ipynb` | Exploratory analysis using SQL (SQLite) — queries across payload mass, launch sites, booster versions, and mission outcomes |
| `5-eda-dataviz.ipynb` | Visual EDA with matplotlib and seaborn — payload vs. orbit, launch site success rates, yearly trends |
| `6-launch-site-location.ipynb` | Interactive geospatial analysis with Folium — maps launch sites, marks successes/failures, calculates distances to nearby infrastructure |
| `7-ml-prediction.ipynb` | Trains and tunes four classifiers (Logistic Regression, SVM, Decision Tree, KNN) with GridSearchCV; evaluates on test set |

---

## Methods

**Data collection:** SpaceX v4 API + BeautifulSoup web scraping

**EDA:** SQL queries against SQLite, matplotlib/seaborn plots, Folium maps

**Modeling pipeline:**
- Standardized features with `StandardScaler`
- 80/20 train-test split
- Hyperparameter tuning via 10-fold `GridSearchCV`
- Models compared: Logistic Regression, SVM, Decision Tree, KNN
- Evaluation: accuracy score + confusion matrix

---

## Key Findings

- Launch success rate has improved substantially over time as SpaceX iterated on the booster design
- Heavier payloads and certain orbit types (ISS, LEO) show higher landing success correlations
- KSC LC-39A had the highest success rate among active launch sites
- All four classifiers performed comparably on the test set. The dataset is relatively small (~90 rows after wrangling), which limits differentiation between models

---

## Stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Folium · BeautifulSoup · SQLite · requests

---

## Certificate

[IBM Data Science Professional Certificate](https://coursera.org/share/699b77e41c42144d67e9c5496453fcce) — Course 10 Applied Data Science Capstone
