# 🚀 Deployment Guide

This guide covers different deployment options for your Customer Churn Prediction system.

## 📋 Deployment Options

### 1. Batch Prediction System ⚡

Best for: Periodic churn analysis (daily/weekly)

#### Local Batch Processing

```python
# deploy/batch_predict.py
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel

def batch_predict(input_path, output_path, model_path):
    """
    Run batch predictions on new customer data
    """
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("Churn_Batch_Prediction") \
        .getOrCreate()
    
    # Load model
    model = RandomForestClassificationModel.load(model_path)
    
    # Load new customer data
    new_customers = spark.read.csv(input_path, header=True, inferSchema=True)
    
    # Make predictions
    predictions = model.transform(new_customers)
    
    # Save results
    predictions.select('customerID', 'prediction', 'probability') \
        .write.csv(output_path, header=True, mode='overwrite')
    
    spark.stop()

# Usage
batch_predict(
    input_path='new_customers.csv',
    output_path='churn_predictions.csv',
    model_path='models/random_forest_model'
)
```

#### Spark Cluster Deployment

```bash
# Submit to Spark cluster
spark-submit \
  --master spark://cluster-master:7077 \
  --deploy-mode cluster \
  --executor-memory 4G \
  --num-executors 10 \
  --executor-cores 2 \
  batch_predict.py
```

### 2. Real-Time API Service 🌐

Best for: On-demand predictions via REST API

#### Flask API Implementation

```python
# deploy/api_service.py
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.linalg import Vectors
import numpy as np

app = Flask(__name__)

# Initialize Spark (once at startup)
spark = SparkSession.builder \
    .appName("Churn_API") \
    .master("local[*]") \
    .getOrCreate()

# Load model (once at startup)
model = RandomForestClassificationModel.load('models/random_forest_model')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a single customer
    
    Expected JSON format:
    {
        "customerID": "1234-ABCD",
        "tenure": 12,
        "MonthlyCharges": 65.50,
        "TotalCharges": 786.00,
        "Contract": "Month-to-month",
        ...
    }
    """
    try:
        # Get customer data from request
        customer_data = request.json
        
        # Create DataFrame
        df = spark.createDataFrame([customer_data])
        
        # Apply same preprocessing as training
        # (You'd need to save and load the preprocessing pipeline)
        
        # Make prediction
        prediction = model.transform(df)
        
        # Extract results
        result = prediction.select('prediction', 'probability').collect()[0]
        churn_prob = float(result['probability'][1])
        
        response = {
            'customerID': customer_data['customerID'],
            'will_churn': bool(result['prediction']),
            'churn_probability': churn_prob,
            'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.4 else 'Low'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

#### Start the API

```bash
python deploy/api_service.py
```

#### Test the API

```bash
# Health check
curl http://localhost:5000/health

# Predict churn
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customerID": "1234-ABCD",
    "tenure": 12,
    "MonthlyCharges": 65.50,
    "TotalCharges": 786.00,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check"
  }'
```

### 3. Cloud Deployment ☁️

#### Option A: AWS EMR (Elastic MapReduce)

```bash
# Upload code to S3
aws s3 cp spark_churn_prediction/ s3://my-bucket/churn-prediction/ --recursive

# Create EMR cluster
aws emr create-cluster \
  --name "Churn Prediction Cluster" \
  --release-label emr-6.10.0 \
  --applications Name=Spark \
  --instance-type m5.xlarge \
  --instance-count 3 \
  --use-default-roles

# Submit job to EMR
aws emr add-steps \
  --cluster-id j-XXXXXXXXXXXXX \
  --steps Type=Spark,Name="Churn Training",\
ActionOnFailure=CONTINUE,\
Args=[s3://my-bucket/churn-prediction/src/5_full_pipeline.py]
```

#### Option B: Azure Databricks

1. Create Databricks workspace
2. Upload notebooks to workspace
3. Create cluster with Spark 3.4+
4. Run notebooks or schedule jobs
5. Use MLflow for model tracking

```python
# In Databricks notebook
import mlflow
import mlflow.spark

# Log model
with mlflow.start_run():
    mlflow.spark.log_model(model, "churn_model")
    mlflow.log_metrics({
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1_score
    })
```

#### Option C: Google Cloud Dataproc

```bash
# Create Dataproc cluster
gcloud dataproc clusters create churn-cluster \
  --region us-central1 \
  --master-machine-type n1-standard-4 \
  --worker-machine-type n1-standard-4 \
  --num-workers 2

# Submit job
gcloud dataproc jobs submit pyspark \
  --cluster churn-cluster \
  --region us-central1 \
  gs://my-bucket/src/5_full_pipeline.py
```

### 4. Docker Containerization 🐳

#### Dockerfile

```dockerfile
# Dockerfile
FROM apache/spark-py:v3.4.1

# Set working directory
WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY src/ ./src/
COPY utils/ ./utils/
COPY models/ ./models/
COPY data/ ./data/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for API
EXPOSE 5000

# Set entrypoint
CMD ["python", "src/5_full_pipeline.py"]
```

#### Build and Run

```bash
# Build Docker image
docker build -t churn-prediction:latest .

# Run container
docker run -p 5000:5000 -v $(pwd)/models:/app/models churn-prediction:latest

# For API service
docker run -p 5000:5000 churn-prediction:latest python deploy/api_service.py
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  spark-master:
    image: bitnami/spark:3.4.1
    environment:
      - SPARK_MODE=master
    ports:
      - '8080:8080'
      - '7077:7077'
  
  spark-worker:
    image: bitnami/spark:3.4.1
    environment:
      - SPARK_MODE=worker
      - SPARK_MASTER_URL=spark://spark-master:7077
    depends_on:
      - spark-master
  
  churn-api:
    build: .
    ports:
      - '5000:5000'
    volumes:
      - ./models:/app/models
    depends_on:
      - spark-master
```

### 5. Monitoring & Maintenance 📊

#### Model Monitoring Script

```python
# monitoring/model_monitor.py
import time
from datetime import datetime
import pandas as pd

def monitor_model_performance(predictions_path, actual_path, threshold=0.05):
    """
    Monitor model performance over time
    Detect model drift
    """
    # Load predictions and actuals
    predictions = pd.read_csv(predictions_path)
    actuals = pd.read_csv(actual_path)
    
    # Calculate accuracy
    correct = (predictions['prediction'] == actuals['actual']).sum()
    accuracy = correct / len(actuals)
    
    # Check for drift
    if accuracy < threshold:
        send_alert(f"Model accuracy dropped to {accuracy:.2%}")
    
    # Log metrics
    log_metrics({
        'timestamp': datetime.now(),
        'accuracy': accuracy,
        'num_predictions': len(predictions)
    })

def send_alert(message):
    """Send alert via email/Slack/etc."""
    print(f"ALERT: {message}")
    # Implement email/Slack notification

def log_metrics(metrics):
    """Log metrics to monitoring system"""
    # Write to database or monitoring service
    pass
```

#### Automated Retraining

```python
# monitoring/auto_retrain.py
from datetime import datetime, timedelta
import schedule

def should_retrain():
    """
    Determine if model should be retrained
    Based on: data drift, performance degradation, time elapsed
    """
    last_train_date = get_last_train_date()
    days_since_training = (datetime.now() - last_train_date).days
    
    # Retrain every 30 days or if performance drops
    return days_since_training >= 30 or model_accuracy() < 0.80

def retrain_model():
    """Retrain model with new data"""
    print("Starting model retraining...")
    # Run full pipeline
    os.system("python src/5_full_pipeline.py")
    print("Retraining complete!")

# Schedule retraining check
schedule.every().day.at("02:00").do(lambda: retrain_model() if should_retrain() else None)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## 🔒 Security Considerations

1. **API Authentication**
```python
from functools import wraps
from flask import request

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != os.environ.get('API_KEY'):
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... prediction code
```

2. **Input Validation**
3. **Rate Limiting**
4. **HTTPS/TLS**
5. **Data Encryption**

## 📈 Scaling Strategies

### Horizontal Scaling
- Add more Spark workers
- Use Kubernetes for auto-scaling
- Implement load balancing

### Vertical Scaling
- Increase executor memory
- Use more CPU cores
- Optimize Spark configuration

### Caching
- Cache frequently used data
- Use Spark's built-in caching
- Implement Redis for API responses

## 🎯 Production Checklist

- [ ] Model versioning implemented
- [ ] API documentation created
- [ ] Monitoring dashboard set up
- [ ] Automated retraining scheduled
- [ ] Backup and recovery plan
- [ ] Security measures in place
- [ ] Load testing completed
- [ ] Logging configured
- [ ] Error handling robust
- [ ] Documentation updated

## 📞 Support

For deployment issues:
1. Check logs in `outputs/` directory
2. Verify Spark configuration
3. Ensure all dependencies installed
4. Review error messages carefully

---

**Remember**: Start with simple deployment (local/batch) and scale up based on needs!
