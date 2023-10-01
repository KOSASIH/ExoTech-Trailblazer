def predict_trajectory(gravity, alignment, fuel):
    factors = np.array([[gravity, alignment, fuel]])
    trajectory = model.predict(factors)
    return trajectory[0]
