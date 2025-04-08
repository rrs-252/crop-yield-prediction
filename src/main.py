import argparse
from predictor import YieldPredictor

def main():
    parser = argparse.ArgumentParser(description='Crop Yield Predictor')
    parser.add_argument('--lat', type=float, required=True)
    parser.add_argument('--lon', type=float, required=True)
    parser.add_argument('--crop', required=True, 
                      choices=['rice','wheat','maize','pearl_millet','finger_millet','barley'])
    
    args = parser.parse_args()
    
    predictor = YieldPredictor()
    try:
        yield_pred = predictor.predict(args.lat, args.lon, args.crop)
        print(f"Predicted {args.crop} yield: {yield_pred:.2f} kg/ha")
    except ValueError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
