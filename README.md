# Food Delivery Time Prediction 🚚

This project is inspired by my work in logistics delivery, where accurately predicting **travel time** is essential to meet customer expectations and **Service Level Agreements (SLA)**.  

The goal is to estimate the **Minimum Travel Time** for deliveries, helping logistics teams plan better and keep customers happy.

To simulate real-world delivery situations, a **food delivery dataset** is used as an example. I experimented with several machine learning models, including **Linear Regression**, **Random Forest**, and **XGBoost**, and optimized them to achieve accurate and reliable predictions.

The project also deploy the model as a public API using **FastAPI**. This allows integration with front-end tools, such as maps or route planners. Users can input delivery details and instantly receive an estimated travel time, making it a practical tool for daily operations.

In the future, I plan to apply model with real delivery data from my company and connect it to the existing **Route Optimization system**.  
The predicted **Minimum Travel Time** can be used as a **constraint** in the optimization algorithm.  
By adding this constraint, the system will be able to create routes that are more accurate and efficient, helping the team improve delivery planning and meet customer expectations.



## Dataset

The dataset is from [Kaggle: Food Delivery Time Prediction](https://www.kaggle.com/datasets/denkuznetz/food-delivery-time-prediction)


## Description

For the submission
- The `notebook.ipynb` (inside the `notebook/` folder) also includes Description about:
  - **Exploratory Data Analysis (EDA)**
  - **Model Training Strategy**
- `script.py` (inside the `script/` folder) allows you to run the model locally and save it as a pickle file.  
- `app.py` serves as the entry point for the **FastAPI** web service, loading the best model and providing prediction endpoints.  
- `Dockerfile` is included to build and deploy the entire application as a container.

This project predicts delivery times based on features such as:

- Distance (`distance_km`)
- Preparation time (`preparation_time_min`)
- Courier experience (`courier_experience_yrs`)
- Weather (`weather`)
- Traffic level (`traffic_level`)
- Time of day (`time_of_day`)
- Vehicle type (`vehicle_type`)

### Model Performance

| Model                | Validation MSE | Validation MAE | Test MSE   | Test MAE |
|---------------------|----------------|----------------|------------|----------|
| Linear Regression    | 28.23          | 3.90           | 35.57      | 4.04     |
| Random Forest        | 28.68          | 3.35           | 15.06      | 2.80     |
| XGBoost              | 0.03           | 0.06           | 0.16       | 0.07     |


## Environment Setup

This project provides a file with all dependencies and uses a **virtual environment** or **UV** to isolate the project packages.  

### 1️⃣ Create a virtual environment

```bash
# Create a virtual environment named 'venv'
python -m venv venv
```
### 2️⃣ Activate the virtual environment

```
# activate environment for bash
source venv/bin/activate 

# or activate environment for cmd windows
venv\Scripts\activate 
```

### 3️⃣ Install dependencies
```
# Install all required dev packages for running script and model locally
pip install -r requirements_dev.txt
```

### Note
- `requirements_dev.txt` → for running model training and development  
- `requirements.txt` → for running **FastAPI** in production with minimal dependencies


## Model API Endpoints
I containerized and deployed the best performing model (**XGBoost**) using **FastAPI** on **Google Cloud** as a public API.  
Endpoints URL: ```https://mtt-ml-767581861887.asia-southeast1.run.app```

#### Predicts **Minimum Travel Time** based on order details.

```http
  POST /predict
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `weather` | `string` | Required One of ['windy', 'clear', 'foggy', 'rainy', 'snowy'] |
| `traffic_level` | `string` | Required One of ['low', 'medium', 'high'] |
| `time_of_day` | `string` | Required One of ['afternoon', 'evening', 'night', 'morning'] |
| `vehicle_type` | `string` | Required One of ['scooter', 'bike', 'car'] |
| `distance_km` | `float` | greater than 0 |
| `preparation_time_min` | `float` | greater than 0 |
| `courier_experience_yrs` | `float` | greater than 0 |

![Test API](test_api.jpg)

#### Check health 

```http
  GET /health
```
Example Response
```json
{
    "status": "ok"
}
```
