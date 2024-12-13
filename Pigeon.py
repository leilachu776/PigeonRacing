# Prepare the data for the machine-learning model

# Drop non-numerical columns and Pigeon_ID (irrelevant for prediction)
features = data.drop(columns=["Pigeon_ID", "Winning_Likelihood", "Wind_Direction", "Weather_Conditions"])
target = data["Winning_Likelihood"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

accuracy, report
