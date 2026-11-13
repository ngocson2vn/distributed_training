# A toy ViT model

## Train
```bash
./run_train.sh
```

## Infer
```bash
python infer.py
```

Output:
```
image 0: deer (class 4, 85.7% confidence)
image 1: bird (class 2, 84.4% confidence)
image 2: truck (class 9, 86.8% confidence)
image 3: automobile (class 1, 55.7% confidence)
image 4: deer (class 4, 87.0% confidence)
image 5: ship (class 8, 67.8% confidence)
image 6: deer (class 4, 88.6% confidence)
image 7: deer (class 4, 57.0% confidence)
image 8: dog (class 5, 33.7% confidence)
image 9: truck (class 9, 22.9% confidence)
image 10: dog (class 5, 45.9% confidence)
image 11: automobile (class 1, 89.9% confidence)
image 12: frog (class 6, 65.0% confidence)
image 13: automobile (class 1, 91.4% confidence)
image 14: cat (class 3, 40.5% confidence)
```