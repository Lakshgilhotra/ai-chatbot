import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import gradio as gr
import difflib

# Load data
file_path = r"C:\Users\laksh\Downloads\HB1 1.csv"
data = pd.read_csv(file_path)

# Map gender to numeric
data['gender'] = data['gender'].map({'male': 0, 'female': 1})

# Convert location to numeric codes
data['location_code'] = data['location'].astype('category').cat.codes

# Encode age ranges to numeric categories
data['age_code'] = data['age'].astype('category').cat.codes

# Ensure the description column is included in the dataset
description_dict = {
    'Mental Health': 'Excellent',
    'Physical Health': 'Excellent',
    'Smoking': 'Smoking daily',
    'Vaping': 'Vaping daily.'
}
data['description'] = data['behavior'].map(description_dict)

# Define features and target variable
X = data[['age_code', 'gender', 'location_code', 'likelihood_percent']]
y = data['behavior']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Define Random Forest model
rf_model = RandomForestClassifier(random_state=42)

# Perform Randomized Search to find the best hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid, 
                                      n_iter=20, cv=5, verbose=1, random_state=42, n_jobs=-1)

# Fit the model
rf_random_search.fit(X_train, y_train)

# Best model from random search
best_rf_model = rf_random_search.best_estimator_

# Initialize state variables
state = {
    "age": None, 
    "gender": None, 
    "location_likelihood_map": {}, 
    "name": None, 
    "action": None, 
    "last_response": None, 
    "last_action": None,
    "awaiting_input": None
}

def correct_location_name(location, location_categories):
    """Corrects the location name based on the closest match from the dataset."""
    close_matches = difflib.get_close_matches(location, location_categories, n=1, cutoff=0.6)
    if close_matches:
        return close_matches[0]
    else:
        raise ValueError(f"Location '{location}' not found in dataset. Available locations are: {location_categories}")

def get_age_code(age_input):
    """Convert age input to age code based on dataset categories."""
    age_categories = data['age'].astype('category').cat.categories.tolist()
    
    # Normalize age categories by removing trailing spaces
    normalized_age_categories = [age.strip() for age in age_categories]
    
    # Convert age_input to integer if it's a single age
    if age_input.isdigit():
        age_input = int(age_input)
        for age_range in normalized_age_categories:
            try:
                start_age, end_age = age_range.split('-')
                if start_age.isdigit() and end_age.isdigit():
                    if age_input in range(int(start_age), int(end_age) + 1):
                        return normalized_age_categories.index(age_range)
            except ValueError:
                continue
        raise ValueError(f"Age '{age_input}' is out of the expected range. Please provide a valid age group or age.")
    else:
        # Check for exact matches with age ranges
        if age_input in normalized_age_categories:
            return normalized_age_categories.index(age_input)
        else:
            raise ValueError(f"Age '{age_input}' is not in the valid age ranges. Please enter an age group (e.g., '25-34').")

# Function to predict likelihood and generate description
def predict_likelihood(age_input, gender, location_likelihood_map):
    # Convert inputs to numeric values
    gender = 0 if gender == 'male' else 1
    
    # Map location and age to numeric codes
    location_categories = data['location'].astype('category').cat.categories.tolist()
    
    age_code = get_age_code(age_input)
    
    # Get the behavior categories
    behavior_categories = data['behavior'].astype('category').cat.categories.tolist()
    
    results = []
    for location, likelihood_percent in location_likelihood_map.items():
        # Correct the location name
        corrected_location = correct_location_name(location, location_categories)
        
        location_code = location_categories.index(corrected_location)
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[age_code, gender, location_code, likelihood_percent]], 
                                  columns=['age_code', 'gender', 'location_code', 'likelihood_percent'])
        
        # Impute and scale features
        scaled_features = scaler.transform(imputer.transform(input_data))
        
        # Get the probabilities for all classes
        probabilities = best_rf_model.predict_proba(scaled_features)[0]
        
        # Find the behavior with the highest probability
        max_prob_index = probabilities.argmax()
        behavior = behavior_categories[max_prob_index]
        likelihood = probabilities[max_prob_index]
        
        # Get description
        description = description_dict.get(behavior, 'No description available')

        # Determine feedback
        feedback = 'Good' if likelihood > 0.6 else ('Fair' if likelihood > 0.4 else 'Bad')

        results.append({
            'Age': age_input,
            'Location': corrected_location,
            'Predicted Behavior': behavior,
            'Probability': likelihood,
            'Condition': description,
            'Gender': 'Male' if gender == 0 else 'Female',
            'Feedback': feedback
        })
    
    return pd.DataFrame(results)

def iterate_input(query):
    query = query.strip().lower()
    
    # Greeting Queries
    if query in ["hi", "hello", "hey"]:
        return "Hello! How can I assist you today? You can type 'predict' to start or 'help' for guidance."

    # Help Command to explain the functionality
    if query == "help":
        return ("Commands available:\n"
                "- 'predict': Start predicting behavior\n"
                "- 'reset': Reset the conversation\n"
                "- 'change age': Change your age input\n"
                "- 'change gender': Change your gender input\n"
                "- 'repeat': Repeat the last response\n"
                "- 'bye': Exit the chatbot\n"
                "- 'encourage me': Get an encouraging message\n"
                "Let's start by typing your name!")

    # Reset confirmation
    if query == "reset":
        return "Are you sure you want to reset? Type 'yes' to confirm or 'no' to continue."

    if query == "yes":
        reset_state()
        return start_prompt()

    if query == "no":
        return "Continuing where we left off. Please provide your next input."

    # Undo Query
    if query == "undo":
        if state["last_action"]:
            # Undo last action based on state
            if state["last_action"] == "age":
                state["age"] = None
                return "Last action undone. Please provide your correct age or age group."
            if state["last_action"] == "gender":
                state["gender"] = None
                return "Last action undone. Please provide your correct gender (male or female)."
        return "There is no action to undo."

    # Change Input Queries
    if query == "change age":
        state["age"] = None
        return "Please provide your new age or age group."

    if query == "change gender":
        state["gender"] = None
        return "Please provide your correct gender (male or female)."

    # Farewell Queries
    if query in ["bye", "goodbye", "exit"]:
        return "Goodbye! Feel free to come back anytime if you need more predictions. Take care!"

    # Encouragement Query
    if query == "encourage me":
        return "You're doing great! Keep going, and never stop learning."

    # Repeat Query
    if query == "repeat":
        return state.get("last_response", "Sorry, I don't have anything to repeat.")

    # Main flow handling for name, age, gender, prediction, etc.
    response = None

    # Name input
    if state["name"] is None:
        state["name"] = query.title()
        response = f"Hello {state['name']}! Welcome to the behavior prediction bot. You can type 'predict' to start or 'help' to know more about available commands."
        state["awaiting_input"] = "predict"
    
    # Predict flow
    if state["awaiting_input"] == "predict" and query == "predict":
        state["awaiting_input"] = "age"
        response = "Let's begin. Can you tell me your age group or specific age?"

    # Age input
    elif state["awaiting_input"] == "age":
        age_input = query.strip()
        try:
            age_code = get_age_code(age_input)
            state["age"] = age_input
            state["awaiting_input"] = "gender"
            response = f"Got it! You're in the {age_input} age group. Now, what's your gender (male or female)?"
        except ValueError as e:
            response = f"{str(e)} Please provide your age group (e.g., '25-34') or a specific age."

    # Gender input
    elif state["awaiting_input"] == "gender":
        gender_input = query.strip().lower()
        if gender_input in ['male', 'female']:
            state["gender"] = gender_input
            state["awaiting_input"] = "location"
            response = f"Got it! You're {gender_input}. Now please provide the location and likelihood percentage (e.g., 'CityA with a likelihood percentage of 0.7')."
        else:
            response = "Invalid input for gender. Please type 'male' or 'female'."

    # Location and likelihood input
    elif state["awaiting_input"] == "location":
        if "with a likelihood percentage of" in query:
            location, likelihood_str = query.split("with a likelihood percentage of")
            location = location.strip()
            try:
                likelihood = float(likelihood_str.strip())
                if 0 <= likelihood <= 100:
                    state["location_likelihood_map"][location] = likelihood
                    response = f"Location '{location}' added with a likelihood of {likelihood:.2f}. Do you want to add another location? If not, type 'predict' to get results."
                    state["awaiting_input"] = "final_predict"
                else:
                    response = "The likelihood percentage must be between 0 and 100. Please enter a valid value."
            except ValueError:
                response = "Couldn't parse likelihood. Please enter a valid number after 'with a likelihood percentage of'."

    # Final prediction
    elif state["awaiting_input"] == "final_predict" and query == "predict":
        if state["location_likelihood_map"]:
            results_df = predict_likelihood(state["age"], state["gender"], state["location_likelihood_map"])
            
            # Reset state after prediction
            reset_state()

            # Show the prediction result
            if not results_df.empty:
                max_prob_row = results_df.loc[results_df['Probability'].idxmax()]
                behavior = max_prob_row['Predicted Behavior']
                condition = max_prob_row['Condition']
                feedback = max_prob_row['Feedback']
                response = f"Based on the data, the most likely behavior is '{behavior}'. Condition: '{condition}'. Feedback: '{feedback}'."
            else:
                response = "No results found. Please try again with valid inputs."
        else:
            response = "Please provide location and likelihood percentage before proceeding."

    # Capture last response for repeat functionality
    state["last_response"] = response
    return response

def reset_state():
    state["name"] = None
    state["age"] = None
    state["gender"] = None
    state["location_likelihood_map"] = {}
    state["action"] = None
    state["last_action"] = None
    state["last_response"] = None
    state["awaiting_input"] = None

def start_prompt():
    return "Hello! How can I help you today? Please provide your name to get started:"

# Define the Gradio interface with updated chatbot
interface = gr.Interface(
    fn=iterate_input,
    inputs=gr.Textbox(label="Input your query"),
    outputs="text",
    title="Behavior Prediction Chatbot",
    description=("This chatbot will guide you step by step to predict behavior based on the provided data. "
                 "Type 'help' for available commands."),
    examples=[["John"], ["predict"], ["25-34"], ["male"], ["CityA with a likelihood percentage of 0.7"], ["predict"]],
    theme="compact"
)

# Launch the interface
interface.launch(share=True)







