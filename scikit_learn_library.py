from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
X_train, y_train = load_preprocessed_data()

# Initialize and train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train, y_train)

# Perform predictions on new data
X_test = preprocess_new_data(new_telescope_data)
predictions = classifier.predict(X_test)
