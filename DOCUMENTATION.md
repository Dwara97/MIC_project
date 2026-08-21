# Documentation — AI Travel Analyst

This file goes into more depth than the README on the technical decisions made throughout the project. The README gives the overview; this is the "why" behind each step, for anyone (including future me) who wants to understand the reasoning without re-reading the whole notebook.

## 1. Dataset Overview

- **Source:** `flight_pricing_dataset.csv` (provided dataset for the challenge)
- **Size:** 100,000 rows, 18 columns
- **Target variable:** `Price`
- **Columns used for analysis/modeling:** Airline, Source, Destination, Duration, Total_Stops, Distance_km, Travel_Class, Days_Before_Departure, Season, Passenger_Count, Price
- **Columns not used:** Flight_ID, Departure_Date, Departure_Time, Arrival_Time, Weekday, Aircraft_Type, Booking_Channel (excluded because they weren't needed for the price-driver analysis this project focuses on, and several had a meaningful chunk of missing data that wasn't worth cleaning for columns not being used)

## 2. Data Cleaning — Detailed Breakdown

### 2.1 Duration → Duration_hrs

**Problem:** Column mixed decimal-hour values (e.g., `1.67`) with text-formatted values (e.g., `"0h 45m"`) in the same column.

**Approach:** Wrote a custom function (`duration_in_dec`) that:
1. Checks for `pd.isna()` first, before any string operations, to avoid crashing on missing values
2. Detects whether the value contains `'h'` (text format) or is already numeric
3. For text format, splits on `'h'` to extract hours and minutes separately, converts minutes to a fraction of an hour, and sums them
4. For already-numeric values, converts directly with `float()`
5. Wraps the numeric conversion in a try/except, returning `None` on failure rather than crashing

**Decision:** Kept the original `Duration` column alongside the new `Duration_hrs` column rather than overwriting it, for traceability — this let me verify the conversion was correct by comparing both columns side by side, and made it possible to recover from a mistake (see section 4).

### 2.2 Total_Stops

**Problem:** Mixed numeric (`0`, `1`) and text (`"non-stop"`, `"1 stop"`) values.

**Approach:** Used `.replace()` with an explicit mapping dictionary, then `pd.to_numeric(errors='coerce')` as a safety net to catch anything not covered by the mapping.

### 2.3 Price

**Problem:** Values formatted as currency strings, e.g., `"Rs. 1,51,632.89"`.

**Approach:** 
1. `.astype(str)` to guarantee string type before string operations
2. `.str.replace('Rs.', '', regex=False)` — `regex=False` is important here since `.` is a special regex character (matches any character) and would have matched more than intended without it
3. `.str.replace(',', '', regex=False)` to remove thousands separators
4. `.str.strip()` to remove leftover whitespace
5. `pd.to_numeric(errors='coerce')` to finalize the conversion

### 2.4 Distance_km, Days_Before_Departure

Same pattern as Price — strip unit suffixes (`"km"`, `"days"`), then convert with `pd.to_numeric(errors='coerce')`.

### 2.5 Passenger_Count

**Problem:** Mixed word-form numbers (`"two"`, `"four"`) with digit values.

**Approach:** Looped through a word-to-digit mapping dictionary, replacing each word before final numeric conversion.

### 2.6 Airline (capitalization)

**Problem:** Same airline appeared under multiple capitalizations — e.g., `"Indigo"`, `"indigo"`, `"INDIGO"` — which pandas treats as distinct categories since string comparison is case-sensitive.

**Approach:** `.str.strip().str.title()` to standardize casing and remove stray whitespace. This directly affected the by-airline visualization — before this fix, the same airline was being split across 2-3 separate boxes in the boxplot.

### 2.7 Handling missing values

**Initial mistake:** Early in the process, missing values were filled with `0`. This was wrong — `0` is not a realistic value for Price, Distance_km, Days_Before_Departure, or Passenger_Count, and would have taught the model false patterns (e.g., that some flights are free). Caught this by noticing the `describe()` output didn't make sense, reloaded the raw CSV, and redid the cleaning pipeline correctly.

**Final approach:** `df.dropna(subset=[...])`, listing only the columns actually needed for analysis and modeling. This avoided dropping rows for missing values in unused columns (like Aircraft_Type), preserving more usable data than a blanket `dropna()` would have.

**Result:** 57,615 of 100,000 rows retained (57.6%).

### 2.8 Key pandas concept used throughout: `errors='coerce'`

`pd.to_numeric(column, errors='coerce')` converts any value that can't be parsed as a number into `NaN`, instead of raising an exception and halting execution (`errors='raise'`, the default). This was essential for cleaning columns with leftover unparseable text (including literal `"nan"` strings) without the pipeline crashing partway through.

## 3. Exploratory Data Analysis

Five required visualizations, plus one supplementary chart:

1. **Price distribution (histogram + KDE)** — revealed a bimodal distribution
2. **Price distribution by Travel_Class (supplementary)** — confirmed the second peak (~₹2,00,000) is driven specifically by Business class fares
3. **Price by Airline (boxplot)** — showed budget carriers (IndiGo, SpiceJet, GoFirst, AirAsia India) clustering low, international carriers (Emirates, Qatar Airways, etc.) clustering higher with wider spread; Air India and Vistara showed a mixed pattern suggesting broader cabin-class coverage
4. **Price by Total_Stops (boxplot)** — counterintuitively, more stops correlated with higher median price, likely because multi-stop flights are more common on longer (and thus more expensive) international routes
5. **Price vs Days_Before_Departure (scatter + aggregated line)** — the raw scatter (57k points) was too dense to read meaningfully, so a `groupby('Days_Before_Departure')['Price'].mean()` aggregation was used to produce a clean trend line, which revealed a sharp price spike for bookings made within ~15 days of departure
6. **Correlation heatmap** — quantified relationships between all numeric features and Price

## 4. Feature Engineering

Categorical columns (Airline, Source, Destination, Travel_Class, Season) were one-hot encoded via `pd.get_dummies(..., drop_first=True)`. `drop_first=True` drops one category per column to avoid redundant/perfectly-collinear columns (if all other category columns are False, the dropped one is implied). This expanded the feature set from 10 original columns to 129.

## 5. Model Choice and Justification

**Model used:** `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)`

**Why Random Forest over alternatives:**
- Handles a mix of numeric and one-hot encoded categorical features without requiring feature scaling
- Less prone to overfitting than a single decision tree, since it averages predictions across 100 independently-built trees, each trained on a random subset of data and features
- Provides feature importances out of the box, which directly supports the "explain key features driving predictions" requirement
- Reasonably robust to outliers compared to linear models, which matters here given the bimodal price distribution

**Why not Linear Regression:** Price's relationship with the features isn't purely linear (e.g., the sharp non-linear spike in the last-15-days booking window, and the two-cluster class-driven price distribution) — a linear model would struggle to capture these patterns.

## 6. Evaluation

**Train/test split:** 80/20, `random_state=42` for reproducibility.

**Metrics:**
| Metric | Value | Interpretation |
|---|---|---|
| MAE | ₹13,854.19 | Average prediction error, treating all errors equally |
| RMSE | ₹46,375.40 | Average error, weighted more heavily toward large mistakes |
| R² | 0.62 | Proportion of price variance explained by the model |

**Why RMSE is much larger than MAE:** This gap indicates most predictions are reasonably close, but a subset are far off, and RMSE's squaring penalizes those large errors disproportionately. The working hypothesis is that Business/First class flights — which form a distinct high-price cluster — are harder for the model to predict precisely than the more numerous, more consistent Economy-range flights.

## 7. Feature Importance

The trained model's `.feature_importances_` were extracted and plotted for the top 15 features. Results were consistent with the correlation heatmap findings from EDA — Distance_km, Duration_hrs, and Travel_Class-related encoded columns ranked among the most important, which served as a useful cross-check that the model learned patterns consistent with the manual analysis.

## 8. Known Limitations

- The model does not separately account for the bimodal price structure caused by Travel_Class, which likely limits accuracy specifically on Business/First class predictions
- No hyperparameter tuning was performed (default `n_estimators=100`); a tuned model or gradient boosting approach may perform better
- First class flights did not show the same distinct price clustering that Business class did, despite both being premium cabins — this asymmetry wasn't fully investigated within the project timeline
- Columns with missing values not used in modeling (Departure_Date, Weekday, Aircraft_Type, Booking_Channel) were left as-is rather than cleaned, since they weren't part of the analysis scope
