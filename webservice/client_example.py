import json

import requests

# API Configuration
API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("Health Check:")
            print(f"  Status: {data['status']}")
            print(f"  Model Loaded: {data['model_loaded']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False


def test_root_endpoint():
    """Test the root endpoint"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print("Root Endpoint:")
            print(f"  Message: {data['message']}")
            return True
        else:
            print(f"Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Root endpoint error: {e}")
        return False


def make_prediction(patient_data):
    """Make a single prediction request"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            print("Prediction Result:")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Risk Probability: {result['probability']:.3f}")
            print(
                f"  Prediction: {'Heart Disease Risk' if result['prediction'] == 1 else 'No Heart Disease Risk'}"
            )
            return result
        else:
            print(f"Prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Prediction error: {e}")
        return None


def make_batch_prediction(patients_data):
    """Make batch predictions"""
    try:
        payload = {"data": patients_data}
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code == 200:
            result = response.json()
            print("Batch Prediction Results:")
            for i, prediction in enumerate(result["predictions"]):
                print(f"  Patient {i + 1}:")
                print(f"    Risk Level: {prediction['risk_level']}")
                print(f"    Risk Probability: {prediction['probability']:.3f}")
                print(
                    f"    Prediction: {'Heart Disease Risk' if prediction['prediction'] == 1 else 'No Heart Disease Risk'}"
                )
            return result
        else:
            print(f"Batch prediction failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return None


def get_model_info():
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        if response.status_code == 200:
            data = response.json()
            print("Model Information:")
            print(f"  Model Type: {data['model_type']}")
            print(f"  Number of Features: {data['n_features']}")
            if isinstance(data["features"], list):
                print(f"  Features: {', '.join(data['features'])}")
            else:
                print(f"  Features: {data['features']}")
            return data
        else:
            print(f"Model info failed: {response.status_code}")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Model info error: {e}")
        return None


def main():
    """Main function to test the API"""
    print("Testing Heart Disease Risk Prediction API\n")

    # Test root endpoint
    print("=" * 50)
    test_root_endpoint()

    # Test health check
    print("\n" + "=" * 50)
    if not test_health_check():
        print("API is not healthy, exiting...")
        return

    # Get model info
    print("\n" + "=" * 50)
    get_model_info()

    # Example patient data for testing
    print("\n" + "=" * 50)
    print("Testing with sample patient data:")

    # Low risk patient example
    low_risk_patient = {
        "age": 35.0,
        "sex": 0.0,  # female
        "cp": 0.0,  # no chest pain
        "trestbps": 120.0,
        "chol": 180.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 170.0,
        "exang": 0.0,
        "oldpeak": 0.0,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 2.0,
    }

    print("\nLow Risk Patient Profile:")
    print(json.dumps(low_risk_patient, indent=2))
    make_prediction(low_risk_patient)

    # High risk patient example
    high_risk_patient = {
        "age": 65.0,
        "sex": 1.0,  # male
        "cp": 3.0,  # severe chest pain
        "trestbps": 160.0,
        "chol": 300.0,
        "fbs": 1.0,
        "restecg": 2.0,
        "thalach": 120.0,
        "exang": 1.0,
        "oldpeak": 3.0,
        "slope": 0.0,
        "ca": 3.0,
        "thal": 3.0,
    }

    print("\nHigh Risk Patient Profile:")
    print(json.dumps(high_risk_patient, indent=2))
    make_prediction(high_risk_patient)

    # Test batch prediction
    print("\n" + "=" * 50)
    print("Testing batch prediction:")

    batch_patients = [low_risk_patient, high_risk_patient]
    make_batch_prediction(batch_patients)

    # Test with the example from your API docs
    print("\n" + "=" * 50)
    print("Testing with API example data:")

    example_patient = {
        "age": 63.0,
        "sex": 1.0,
        "cp": 3.0,
        "trestbps": 145.0,
        "chol": 233.0,
        "fbs": 1.0,
        "restecg": 0.0,
        "thalach": 150.0,
        "exang": 0.0,
        "oldpeak": 2.3,
        "slope": 0.0,
        "ca": 0.0,
        "thal": 1.0,
    }

    print("\nExample Patient Profile:")
    print(json.dumps(example_patient, indent=2))
    make_prediction(example_patient)

    # Test with invalid data
    print("\n" + "=" * 50)
    print("Testing with invalid data:")

    invalid_patient = {
        "age": "invalid",  # Invalid type
        "sex": 0.0,
        "cp": 0.0,
        "trestbps": 120.0,
        "chol": 180.0,
        "fbs": 0.0,
        "restecg": 0.0,
        "thalach": 170.0,
        "exang": 0.0,
        "oldpeak": 0.0,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 2.0,
    }

    make_prediction(invalid_patient)

    print("\n" + "=" * 50)
    print("API testing completed!")


def interactive_prediction():
    """Interactive prediction function"""
    print("\nInteractive Prediction Mode")
    print("Enter patient data (or 'quit' to exit):")

    while True:
        try:
            print("\nEnter patient data:")
            age = input("Age: ")
            if age.lower() == "quit":
                break

            sex = input("Sex (0=female, 1=male): ")
            cp = input("Chest pain type (0-3): ")
            trestbps = input("Resting blood pressure: ")
            chol = input("Cholesterol: ")
            fbs = input("Fasting blood sugar > 120 mg/dl (0=no, 1=yes): ")
            restecg = input("Resting ECG (0-2): ")
            thalach = input("Max heart rate: ")
            exang = input("Exercise induced angina (0=no, 1=yes): ")
            oldpeak = input("ST depression: ")
            slope = input("Slope (0-2): ")
            ca = input("Major vessels (0-4): ")
            thal = input("Thalassemia (0-3): ")

            patient_data = {
                "age": float(age),
                "sex": float(sex),
                "cp": float(cp),
                "trestbps": float(trestbps),
                "chol": float(chol),
                "fbs": float(fbs),
                "restecg": float(restecg),
                "thalach": float(thalach),
                "exang": float(exang),
                "oldpeak": float(oldpeak),
                "slope": float(slope),
                "ca": float(ca),
                "thal": float(thal),
            }

            print("\nMaking prediction...")
            result = make_prediction(patient_data)

            if result:
                print(f"\nRESULT: {result['risk_level']} risk of heart disease")
                print(f"Probability: {result['probability']:.1%}")

        except ValueError:
            print("Invalid input. Please enter numeric values.")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_prediction()
    else:
        main()
