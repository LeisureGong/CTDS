# Local Subsequence-based Distribution for Time Series Clustering



## Usage
For datasets less than 150 in length:
```shell
python main.py --dataset TwoLeadECG --type 1 --d 40
python main.py --dataset CBF --type 1 --d 60
```

For datasets greater than 150 in length:
```shell
python main.py --dataset FaceFour --type 0 --d 60
python main.py --dataset StarLightCurves --type 0 --d 72
```





## Dependencies
- numpy
- scipy
- joblib
- tslearn
- scikit-learn
